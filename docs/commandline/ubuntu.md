---
layout: basic
---

## Some commands that are specific to Ubuntu/Linux

### Changing Java version

can try

```
sudo update-alternatives --config java
```

See [this link](https://aboullaite.me/switching-between-java-versions-on-ubuntu-linux/)  

NOTE:  Need to run ```java -version``` and ```echo $JAVA_HOME``` to verify the above command switched java.  If it didn't work then add the following to the ```.bashrc``` file.  

(below is an example, location of java will be different on different machines)

```
export JAVA_HOME='/usr/lib/jvm/java-11-openjdk-amd64'
````