//imagej-macro "cellSegmentation" (Herbie G., 18. June 2025)
/*
Requires two ImageJ-plugins
"MorphoLibJ_-1.6.4.jar"
<https://github.com/ijpb/MorphoLibJ/releases>
&
"Adjustable_Watershed.class"
<https://github.com/imagej/imagej.github.io/blob/main/media/adjustable-watershed/Adjustable_Watershed.class>
*/
if (nSlices!=1||bitDepth()!=16) exit("requires a single 16bit image!");
setBatchMode(true);
run("Duplicate...","title=cpy");
run("Gaussian Blur...","sigma=2");
run("Subtract Background...","rolling=2 sliding");
setAutoThreshold("Huang dark no-reset");
run("Analyze Particles...","size=500-Infinity show=Masks");
run("Invert LUT");
run("Adjustable Watershed","tolerance=8");
run("Connected Components Labeling","connectivity=4 type=[16 bits]");
setBatchMode(false);
exit();
//imagej-macro "cellSegmentation" (Herbie G., 18. June 2025)


