

## JavaCPP Hints

When using JavaCpp to wrap c++ code we use an (optional) parse step and build step.  

## Parse step

It's common to see the an error during the parse step.  Making a trivial change to the java class that is being parsed seems to trigger the parser and fix the error. 

```
[ERROR] Failed to execute goal org.bytedeco:javacpp:1.5.2:parse (generate-sources) 
```
## Build step
Second error you may hit is on the build step.  The problem in this case is often that maven cannot find source files in the src/gen directory.

Need to use the maven build helper plugin to add this directory. 

```
[ERROR] Failed to execute goal org.bytedeco:javacpp:1.5.2:build (process-classes) 
```
