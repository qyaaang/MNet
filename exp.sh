#!/bin/bash
# "2dof" "2dof_fem" "3dof" "3dof_beam" "4dof" "10dof" "truss"
model="2dof_fem"
optimizers=("GD" "Adam")
norms=(1 2)
for optimizer in "${optimizers[@]}"; do
  # Learning rate
  if [ "$optimizer" == "GD" ]; then
    lr=1.
  else
    lr=0.1
  fi
  for norm in "${norms[@]}"; do
    # Tolerance
    if [ "$norm" == 1 ]; then
      tol=1e-10
    else
      tol=1e-20
    fi
    # Training
    printf "\033[1;32mModel: %s\n\033[0m" "$model"
    printf "\033[1;32mNorm: %s\tOptimizer: %s\t\n\033[0m" "$norm" "$optimizer"
    if [ "$model" == "2dof" ]; then
      python3 exp_2dof.py  --norm "$norm" --tol "$tol" --optimizer "$optimizer" --lr "$lr"
    elif [ "$model" == "2dof_fem" ]; then
      python3 exp_2dof_fem.py  --norm "$norm" --tol "$tol" --optimizer "$optimizer" --lr "$lr"
    elif [ "$model" == "3dof" ]; then
      python3 exp_3dof.py  --norm "$norm" --tol "$tol" --optimizer "$optimizer" --lr "$lr"
    elif [ "$model" == "3dof_beam" ]; then
      python3 exp_3dof_beam.py  --norm "$norm" --tol "$tol" --optimizer "$optimizer" --lr "$lr"
    elif [ "$model" == "4dof" ]; then
      python3 exp_mdof.py  --num_dof 4 --norm "$norm" --tol "$tol" --optimizer "$optimizer" --lr "$lr"
    elif [ "$model" == "10dof" ]; then
      python3 exp_mdof.py  --num_dof 10 --norm "$norm" --tol "$tol" --optimizer "$optimizer" --lr "$lr"
    elif [ "$model" == "truss" ]; then
      python3 exp_truss.py  --norm "$norm" --tol "$tol" --optimizer "$optimizer" --lr "$lr"
    fi
  done
done
