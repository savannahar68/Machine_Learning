In support vector machine we try to draw a like which in its best way will divide the dataset without partiality<br/>
The line will be drawn on the distance from each point and then the best fitting line would be chosen <br/>
For prediction whatever to the left of the line falls into that subgroup similarly for right side of line <br/>


There are types of Kernel each supporting a different category <br/>
1.Linear<br/>
2.RBF<br/>
3.Polynomial<br/>


We'll also build SVM from scratch using Vectors.
<br/>
Kernel - It is a function which takes inputs and outputs there similarities.<br/>
		So in real world not all the data we get are linear in nature, so finding a hyperplane seems to be impossible task.But no wait we can convert the data to any dimension using our machines(Lit stuff happening in).
		So we make our data to higher dimensions and then plot our hyperplace accordingly.Kernels are used for this purpose.Disadvantage is processing increases a lot.So to overcome this kernels do a inner product