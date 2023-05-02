# Smipsons_classification

<h2>Result from kagle</h2>

<img src = "./resul_acc.png">
<h2>The fusing of batch and convolution operations</h2>
<p>The main one aim here is to add batch norm operation to convolution operation<br>
It's posible with FC lib<br></p>
<a href = "https://pytorch.org/tutorials/intermediate/custom_function_conv_bn_tutorial.html">Href to the FC documentation</a>
<h3> FC - it's mean FusedConv</h3>
<p>All, what I need is to add special class into my own code</p>
<img src = "./argm.png">
<p>The new one class also going to use standart convolution arguments</p>
<p>But there are no as aarguments, as padding or starait</p>
<img src = "./Module_with_FC.png">