#!/bin/bash

python --version 

EPOCHS=12000
cd Examples/

echo "##################################################################"
echo "############# Running Lorenz Example               ##############"
echo "##################################################################"

python Example_Lorenz.py --model_hypothesis no_hypos --epochs $EPOCHS
echo ""
python Example_Lorenz.py --model_hypothesis globalstability --epochs $EPOCHS
echo ""

echo "##################################################################"
echo "############# Running MHD Example               ##############"
echo "##################################################################"


python Example_MHD.py --model_hypothesis no_hypos --epochs $EPOCHS
echo ""
python Example_MHD.py --model_hypothesis globalstability --epochs $EPOCHS
echo ""

echo "##################################################################"
echo "############# Running Chafee Example               ##############"
echo "##################################################################"

python Example_Chafee_Stability.py --model_hypothesis no_hypos --epochs $EPOCHS
echo ""
python Example_Chafee_Stability.py --model_hypothesis localstability --epochs $EPOCHS
echo ""
python Example_Chafee_Stability.py --model_hypothesis globalstability --epochs $EPOCHS
echo ""

# jupyter nbconvert --execute --to notebook --inplace plotting/Plotting_Chafee.ipynb

echo "##################################################################"
echo "############# Running Burgers Example with Dirichlet #############"
echo "##################################################################"

python Example_Burgers_Dirichilet_Stability_order.py --model_hypothesis no_hypos --epochs $EPOCHS > ./Results/burgers_dirichlet_order_test.txt
echo ""
python Example_Burgers_Dirichilet_Stability_order.py --model_hypothesis localstability --epochs $EPOCHS >> ./Results/burgers_dirichlet_order_test.txt
echo ""
python Example_Burgers_Dirichilet_Stability_order.py --model_hypothesis globalstability --epochs $EPOCHS >> ./Results/burgers_dirichlet_order_test.txt
echo ""

python Example_Burgers_Dirichilet_Stability_reg.py --model_hypothesis no_hypos --epochs $EPOCHS > ./Results/burgers_dirichlet_reg_test.txt
echo ""
python Example_Burgers_Dirichilet_Stability_reg.py --model_hypothesis localstability --epochs $EPOCHS >> ./Results/burgers_dirichlet_reg_test.txt
echo ""
python Example_Burgers_Dirichilet_Stability_reg.py --model_hypothesis globalstability --epochs $EPOCHS >> ./Results/burgers_dirichlet_reg_test.txt
echo ""

# jupyter nbconvert --execute --to notebook --inplace Plotting_Burgers_Dirichilet_orders.ipynb
# jupyter nbconvert --execute --to notebook --inplace Plotting_Burgers_Dirichilet_reg.ipynb

echo "##################################################################"
echo "############# Running Burgers Example with Neumann   #############"
echo "##################################################################"
python Example_Burgers_Neumann_Stability_order.py --model_hypothesis no_hypos --epochs $EPOCHS > ./Results/burgers_neumann_order_test.txt
echo ""
python Example_Burgers_Neumann_Stability_order.py --model_hypothesis localstability --epochs $EPOCHS >> ./Results/burgers_neumann_order_test.txt
echo ""
python Example_Burgers_Neumann_Stability_order.py --model_hypothesis globalstability --epochs $EPOCHS >> ./Results/burgers_neumann_order_test.txt
echo ""

python Example_Burgers_Neumann_Stability_reg.py --model_hypothesis no_hypos --epochs $EPOCHS > ./Results/burgers_neumann_reg_test.txt
echo ""
python Example_Burgers_Neumann_Stability_reg.py --model_hypothesis localstability --epochs $EPOCHS >> ./Results/burgers_neumann_reg_test.txt
echo ""
python Example_Burgers_Neumann_Stability_reg.py --model_hypothesis globalstability --epochs $EPOCHS >> ./Results/burgers_neumann_reg_test.txt
echo ""

# jupyter nbconvert --execute --to notebook --inplace Plotting_Burgers_Neumann_orders.ipynb



