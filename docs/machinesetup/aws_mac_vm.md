## Setting up Mac VMs on AWS  

See https://aws.amazon.com/getting-started/hands-on/launch-connect-to-amazon-ec2-mac-instance/  

1.  Allocate a mac1 or mac2 dedicated host

Note mac1 for intel mac, and mac2 for arm mac. 

a) On EC2 main page select 'dedicated hosts' in the panel on the left. 
b) Allocate Dedicated Host
c) For instance family select mac1 (x86) or mac2 (arm)
d) Make sure you name your dedicated host appropriately so you remember whether it is mac1 or mac2. 

2.  Launch instance onto dedicated host

a) On EC2 main page select 'dedicated hosts' in the panel on the left
b) Select a dedicated host (make sure you select a mac1 host if wanting an x8 mac or a mac2 host for arm Mac)
c) After selecting a host choose Actions->Launch Instance(s) onto host.

to get into the command line interface to the machine with appropriate key pair and public IP.  For example with keypair ```keypair.pem``` and public ip ``ec2-54-90-225-128.compute-1.amazonaws.com````

ssh -i keypair.pem ec2-user@ec2-54-90-225-128.compute-1.amazonaws.com