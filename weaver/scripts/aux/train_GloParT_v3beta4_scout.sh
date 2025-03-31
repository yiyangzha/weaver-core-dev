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
t_qcd:./datasets/20250313_ak8_scouting/QCD_PT-mixed_TuneCP5_13p6TeV_pythia8/*.root \
t_h0hphm:./datasets/20250313_ak8_scouting/H0HpHm_mixed/*.root \
--data-test \
qcd300to470_0:./datasets/20250313_ak8_scouting/infer/QCD_PT-300to470_TuneCP5_13p6TeV_pythia8/*.root \
qcd470to600_0:./datasets/20250313_ak8_scouting/infer/QCD_PT-Low_mixed/*.root \
qcd470to600_1:./datasets/20250313_ak8_scouting/infer/QCD_PT-Low_plus/*.root \
qcd470to600_2:./datasets/20250313_ak8_scouting/infer/QCD_PT-470to600_TuneCP5_13p6TeV_pythia8/*.root \
qcd600to800_0:./datasets/20250313_ak8_scouting/infer/QCD_PT-600to800_TuneCP5_13p6TeV_pythia8/*.root \
qcd1000to1400_0:./datasets/20250313_ak8_scouting/infer/QCD_PT-High_mixed/*.root \
qcd1000to1400_1:./datasets/20250313_ak8_scouting/infer/QCD_PT-High_plus/*.root \
highi:./datasets/20250313_ak8_scouting/infer/Hig_High_mixed/*.root \
higlo:./datasets/20250313_ak8_scouting/infer/Hig_Low_mixed/*.root \
hphmhi:./datasets/20250313_ak8_scouting/infer/HpHm_High_mixed/*.root \
hphmlo:./datasets/20250313_ak8_scouting/infer/HpHm_Low_mixed/*.root \
2024winter_qcdlo:./datasets/20250313_ak8_scouting/infer/2024winter/QCD_PT_Low_TuneCP5_13p6TeV_pythia8/*.root \
2024winter_qcdhi:./datasets/20250313_ak8_scouting/infer/2024winter/QCD_PT_High_TuneCP5_13p6TeV_pythia8/*.root \
--samples-per-epoch $((1500 * 512 / $NGPUS)) --samples-per-epoch-val $((100 * 512)) \
--data-config ${config} \
--model-prefix model/${PREFIX}/net \
--predict-output predict/$PREFIX/pred.root "

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
