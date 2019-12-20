# Reproducing Experiments with Structured Mechanical Models

## Glossary

`Naive` refers to BBNN in the paper.
`ControlAffine` refers to SMM-C in the paper.

## Pre-requisites

1. Install [GNU Parallel](https://www.gnu.org/software/parallel/).

On Ubuntu it should be as easy as:
```
sudo apt install parallel
```

2. Setup julia project
```
julia --project --color=yes -e 'using Pkg; Pkg.instantiate(); Pkg.build(); Pkg.precompile()'
```

3. Dump validation data

```
parallel -j4 ./dumpvaliddata.jl ::: furuta cartpole acrobot doublecartpole ::: "runs/"
```

## Training

### Furuta Pendulum

```
parallel -j5 ./train.jl \
    ::: "ControlAffine" "Naive" \
    ::: furuta \
    ::: 8192 4096 2048 1024 512 256 \
    ::: 42 1234 1339 178969 1234321 \
    ::: 2000 \
    ::: 256 \
    ::: "runs/"
```

```
parallel -j5 ./train.jl \
    ::: "Naive" \
    ::: furuta \
    ::: 16384 32768 \
    ::: 42 1234 1339 178969 1234321 \
    ::: 2000 \
    ::: 256 \
    ::: "runs/"
```


### Cartpole

```
parallel -j5 ./train.jl \
    ::: "ControlAffine" "Naive" \
    ::: cartpole \
    ::: 8192 4096 2048 1024 512 256 \
    ::: 42 1234 1339 178969 1234321 \
    ::: 2000 \
    ::: 256 \
    ::: "runs/"
```

```
parallel -j5 ./train.jl \
    ::: "Naive" \
    ::: cartpole \
    ::: 16384 32768 \
    ::: 42 1234 1339 178969 1234321 \
    ::: 2000 \
    ::: 256 \
    ::: "runs/"
```


### Acrobot

```
parallel -j5 ./train.jl \
    ::: "ControlAffine" "Naive" \
    ::: acrobot \
    ::: 8192 4096 2048 1024 512 256 \
    ::: 42 1234 1339 178969 1234321 \
    ::: 2000 \
    ::: 256 \
    ::: "runs/"
```

```
parallel -j5 ./train.jl \
    ::: "Naive" \
    ::: acrobot \
    ::: 16384 32768 \
    ::: 42 1234 1339 178969 1234321 \
    ::: 2000 \
    ::: 256 \
    ::: "runs/"
```


### Double Cartpole

```
parallel -j5 ./train.jl \
    ::: "ControlAffine" \
    ::: doublecartpole \
    ::: 65536 16384 8192 4096 2048 \
    ::: 42 1234 1339 178969 1234321 \
    ::: 2000 \
    ::: 256 \
    ::: "runs/"
```

```
parallel -j5 ./train.jl \
    ::: "Naive" \
    ::: doublecartpole \
    ::: 262144 131072 65536 32768 16384 8192 4096 2048 \
    ::: 42 1234 1339 178969 1234321 \
    ::: 2000 \
    ::: 256 \
    ::: "runs/"
```


## Evaluation

### Run Evaluation

#### Get best Validation Error from logs
```
./scripts/get_best_validation_error.py <path to logs> > val_error.json
```

#### Get the control costs
```
./evaluate.jl <Naive or ControlAffine> <taskname> cost <path to `best.th` generated from training>
```

### Collect results

```
./scripts/collect_val_error.py <path containing dumped val_error.json>
```

```
./scripts/collect_control_costs.py <path containing dumped cost.json>
```
