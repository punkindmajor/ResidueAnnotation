#main.ipynb
The main.ipynb contains a comprehesive step by step execution for the project
Model Structure, training epoch can be modified in the hyper and pathing block
After training, the last line of the block outputs the model name, which can be copied and paste to replace both model name in plot.ipynb and test.ipynb
#plot.ipynb
Replace MODELNAME in the first block with the output of main,
execute the program should yield the plot associate with the training,
and the output of last block shows the optimal early stopping points under different criteria
#test.ipynb 
Replace MODELNAME in the first block with the output of main, and iteration with the optimal stopping points
execute the program should plot the ROC curve and also report the evaluation metric on testing data
