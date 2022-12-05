# tempPublicRiverTwin
River Twin files to be shared - This is a shortened and cleared up repository for explaining the project to Digital Research UoN

The two sections of the model are split into two seperate files with a YYYY_MM_DD_Runfile to call them. 

Train_RiverTwinWaterMask is the important script that I am struggling to run. 

RiverTwinPartTwo is kept for context. 

Some of Train_RiverTwin is hardcoded at the moment due to errors. 

The function has the ability to create tiles or to use existing temporary tiles. At the moment I am attempting to get it to run on the existing temporary tiles. 

The original issue: https://stackoverflow.com/questions/74643697/tensorflow-datasets-gpu-memory?noredirect=1#comment131753470_74643697
is best explained here. 

I have used a depreciated tf.compat.v1.Session as sess: code which seems to have utilised the GPU but is causing other errors such as x: 
