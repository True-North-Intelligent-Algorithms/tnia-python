//imagej-macro "cellSegmentationSqrt_v0" (Herbie G., 20. June 2025)
/*
Requires two ImageJ-plugins
"MorphoLibJ_-1.6.4.jar"
<https://github.com/ijpb/MorphoLibJ/releases>
&
"Adjustable_Watershed.class"
<https://github.com/imagej/imagej.github.io/blob/main/media/adjustable-watershed/Adjustable_Watershed.class>
*/
if (nSlices!=1||bitDepth()!=16) exit("Requires a single 16bit image!");
setBatchMode(true);
run("Duplicate...","title=cpy");
run("32-bit");
run("Square Root"); //run("Log");
run("Gaussian Blur...","sigma=1.0");
run("Subtract Background...","rolling=200 sliding");
setAutoThreshold("Huang dark");
run("Analyze Particles...","size=500-Infinity show=Masks");
run("Invert LUT");
run("Adjustable Watershed","tolerance=10");
run("Connected Components Labeling","connectivity=8 type=[8 bits]");
setBatchMode(false);
exit();
//imagej-macro "cellSegmentationSqrt_v0" (Herbie G., 20. June 2025)


