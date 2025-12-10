#!/bin/bash -x

RUN=$1
GPUS=$2

if [ -z $GPUS ]; then
    echo "Usage: $0 <ngpu>"
    exit 1
fi
NGPUS=$(echo $GPUS | tr "," "\n" | wc -l)

cmdlineopts="${@:3}"

current_dir=`pwd`
if [[ "$current_dir" != *"weaver-core/weaver" ]]; then
    echo "Please run this script from the weaver directory"
    exit 1
fi

# use final GloParT3 settings (v3beta4 default command w/ num_layers=10, reg_kw:as_resid_of=[1])
## remember: remove all single-quote characters
ARG="--run-mode train --train-mode hybrid \
-o num_nodes 46 -o num_cls_nodes 22 -o use_swiglu_config True -o use_pair_norm_config True \
-o fc_params [(2048,0.1)] -o embed_dims [256,1024,256] -o pair_embed_dims [64,64,64] -o num_heads 16 -o num_layers 10 \
-o reg_kw {'gamma':5.,'composed_split_reg':[True,False],'as_resid_of':[1]} \
--use-amp --batch-size 512 --start-lr 7e-4 --num-epochs 30 --optimizer ranger \
--num-workers 8 --fetch-step 1. --data-split-num 20 \
--network-config networks/stage3/example_GloParT3_forScouting.py \
--data-train \
mixed23:./datasets/20250313_ak8_scouting/2023/mixed2023/*.root \
mixed22:./datasets/20250313_ak8_scouting/2022/mixed2022/*.root \
--data-test \
higlo23BPix:./datasets/20250313_ak8_scouting/2023BPix/BulkGravitonToHHTo4QGluLTau_MH-125_LowPt/*.root \
qcd470to60023BPix:./datasets/20250313_ak8_scouting/2023BPix/infer/QCD_PT-470to600_TuneCP5_13p6TeV_pythia8/*.root \
hphmlo23BPix:./datasets/20250313_ak8_scouting/2023BPix/H3ToHpHmTo4Q_MH-80_LowPt/*.root \
higlo22EE:./datasets/20250313_ak8_scouting/2022EE/BulkGravitonToHHTo4QGluLTau_MH-125_LowPt/*.root \
qcd470to60022EE:./datasets/20250313_ak8_scouting/2022EE/infer/QCD_PT-470to600_TuneCP5_13p6TeV_pythia8/*.root \
hphmlo22EE:./datasets/20250313_ak8_scouting/2022EE/H3ToHpHmTo4Q_MH-80_LowPt/*.root \
--samples-per-epoch $((1500 * 512 / $NGPUS)) --samples-per-epoch-val $((100 * 512)) \
--data-config ${config} \
--model-prefix model/${PREFIX}/net \
--predict-output predict/$PREFIX/pred.root "

#test samples
#higlo23:./datasets/20250313_ak8_scouting/2023/BulkGravitonToHHTo4QGluLTau_MH-125_LowPt/*.root \
#hphmlo23:./datasets/20250313_ak8_scouting/2023/H3ToHpHmTo4Q_MH-80_LowPt/*.root \
#qcdlo23:./datasets/20250313_ak8_scouting/2023/infer/QCD_PT-470to600_TuneCP5_13p6TeV_pythia8/*.root \
#higlo22:./datasets/20250313_ak8_scouting/2022/BulkGravitonToHHTo4QGluLTau_MH-125_LowPt/*.root \
#hphmlo22:./datasets/20250313_ak8_scouting/2022/H3ToHpHmTo4Q_MH-80_LowPt/*.root \
#qcdlo22:./datasets/20250313_ak8_scouting/2022/infer/QCD_PT-470to600_TuneCP5_13p6TeV_pythia8/*.root \
#cmssw_test:/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/GloParT/DNNtuple/signal/cmssw_BulkGravitonToHHTo4QGluLTau_MH-125_LowPt.root \
#qcd300to47022:./datasets/20250313_ak8_scouting/2022/infer/QCD_PT-300to470_TuneCP5_13p6TeV_pythia8/*.root \
#qcd470to60022:./datasets/20250313_ak8_scouting/2022/infer/QCD_PT-470to600_TuneCP5_13p6TeV_pythia8/*.root \
#qcd600to80022:./datasets/20250313_ak8_scouting/2022/infer/QCD_PT-600to800_TuneCP5_13p6TeV_pythia8/*.root \
#qcd300to47023:./datasets/20250313_ak8_scouting/2023/infer/QCD_PT-300to470_TuneCP5_13p6TeV_pythia8/*.root \
#qcd470to60023:./datasets/20250313_ak8_scouting/2023/infer/QCD_PT-470to600_TuneCP5_13p6TeV_pythia8/*.root \
#qcd600to80023:./datasets/20250313_ak8_scouting/2023/infer/QCD_PT-600to800_TuneCP5_13p6TeV_pythia8/*.root \

if [ $RUN == "dryrun" ]; then
    echo "Dryrun mode"
elif [ $RUN == "run" ] || [ $RUN == "autorecover" ]; then
    ARG="$ARG --log-file logs/${PREFIX}/train.log --tensorboard _${PREFIX} "
else
    exit 1
fi

if [ $GPUS == "cpu" ]; then
    cmd="python train.py $ARG $cmdlineopts "
elif [ $GPUS -eq $GPUS 2>/dev/null ]; then
    # if GPUS is an integer
    unset CUDA_VISIBLE_DEVICES
    cmd="python train.py --gpus $GPUS $ARG $cmdlineopts "
else
    # GPU list is separated by comma
    export CUDA_VISIBLE_DEVICES=$GPUS
    cmd="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS train.py --backend nccl $ARG $cmdlineopts "
fi

echo Run command: $cmd

if [ $RUN == "dryrun" ] || [ $RUN == "run" ]; then
    $cmd
elif [ $RUN == "autorecover" ]; then
    epochopts=""
    # if the training is halted, resume from the last epoch
    while true; do
        $cmd $epochopts
        ret=$?
        if [ $ret -eq 0 ]; then
            break
        fi
        echo "Error: return code $ret"
        # match model/${PREFIX}/net_epoch-(\d+)_state.pt and extract the maximum epoch number
        maxepoch=$(ls model/${PREFIX}/net_epoch-*.pt | sed -n 's/.*net_epoch-\([0-9]*\)_state.pt/\1/p' | sort -n | tail -n 1)
        if [ -z $maxepoch ]; then
            epochopts=""
        else
            epochopts="--load-epoch $maxepoch"
            echo "Resuming from epoch $maxepoch"
        fi
        sleep 10
    done
fi
