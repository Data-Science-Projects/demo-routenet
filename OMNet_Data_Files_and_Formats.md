# OMNeT++ Data Files and Formats

The code in the `routenet.data_utils` package processes network data into formats suitable for 
modelling and predictions using TensorFlow. The source files and formats of the data are described 
below. This explanation is based on the 
[KDN datasets_v0 format](https://github.com/knowledgedefinednetworking/NetworkModelingDatasets/tree/master/datasets_v0).

The v0 and [v1 format](https://github.com/knowledgedefinednetworking/NetworkModelingDatasets/tree/master/datasets_v1) 
differ in that the fifth field is: 

```
5.- Average per-packet neperian logarithm of the delay over the packets transmitted in each source-destination pair.

   avg_ln_delay[src_node][dst_node] = n*n*3 + (src_node∗n+dst_node)*8 + 1
```

TODO fill in more details for V1.

There are 
[v0](https://github.com/knowledgedefinednetworking/NetworkModelingDatasets/tree/master/datasets_v0) 
and
[v1](https://github.com/knowledgedefinednetworking/NetworkModelingDatasets/tree/master/datasets_v1) 
datasets. The data sets are bundled into a single tar.gz file for ease of distribution, which 
is expanded into a directory with the name of the network topology.

## graph_attr.txt

This file is part of the data sets, but is not actually used for any purpose. 

## OMNeT++ NED File

Within a give network directory, there is a *.ned file, which describes the network topology, 
including the link capacities. This file is used with [OMNeT++](https://omnetpp.org) to import the 
topology and simulate the network. How this works is not in the public domain.

The example contents below are from the 
[Network_nsfnetbw.ned](tests/unit-resources/nsfnetbw/Network_nsfnetbw.ned) file:

```
package netsimulator;
network  Network_nsfnetbw
{
    parameters:
        int numNodes = 14;
        int numTx = 14;
    types:
        channel Channel10kbps extends ned.DatarateChannel
        {
            delay = 0ms;
            datarate = 10kbps;
        }
        channel Channel40kbps extends ned.DatarateChannel
        {
            delay = 0ms;
            datarate = 40kbps;
        }
    submodules:
        statistic: Statistic {
            numTx = numTx;
            numNodes = numNodes; 
        } 
        node0: Server {
            id = 0;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[3];
        }
        ...     
        node13: Server {
            id = 13;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[2];
        }
        tController: NetTrafficController {
            numTx = numTx;
            numNodes = numNodes; 
            gates:
                out[numTx];
         }

    connections:

      node0.port[0] <--> Channel10kbps <--> node1.port[0];
      ...
      node11.port[2] <--> Channel10kbps <--> node12.port[2];


      tController.out[0] --> { @display("ls=grey,1,d"); } --> node0.tControl;
      ...
      tController.out[13] --> { @display("ls=grey,1,d"); } --> node13.tControl;
}
```
## Simulation Results

The simulation results, for delay, jitter and similar, are contained in multiple further .tar.gz
files, within the same directory as the NED file. Each results file contains samples from 
simulations of network scenarios with a particular routing scheme (TODO defined how?) and a given 
traffic intensity (TODO defined how?). 

The results filenames have this pattern:

```      
results_<topology_name>_<lambda>_<routing_scheme>.tar.gz
```

Where \<lambda> represents the traffic intensity and <routing_scheme> is an identifier of the 
routing configuration used in the simulation. TODO, where/how is this routing configuration defined?

Each of the results tar.gz files contains further files as discussed below.

### Routing.txt

This matrix contains the routing scheme used in the simulations as a destination-based Routing 
Information Base (RIB) at each node, such that:

```
Routing_matrix(src node, dst node) = output port to reach the dst node from the src node
```

Note that the diagonal of the matrix is -1. 

The example contents below are from the 
[Routing.txt](tests/unit-resources/nsfnetbw/Routing.txt) file corresponding to the
[Network_nsfnetbw.ned](tests/unit-resources/nsfnetbw/Network_nsfnetbw.ned) file:
```
-1,0,2,1,1,2,1,0,1,1,0,1,2,2,
0,-1,1,0,2,1,2,2,0,2,2,2,1,1,
0,1,-1,0,2,2,2,1,0,2,2,2,2,2,
0,0,0,-1,1,1,1,1,2,2,2,2,2,1,
0,2,1,0,-1,1,2,2,0,1,1,0,1,1,
0,0,0,1,1,-1,1,1,1,2,3,2,2,3,
0,1,0,0,0,0,-1,1,0,1,1,1,0,0,
0,0,0,1,1,1,1,-1,2,2,2,2,2,2,
0,0,0,0,0,2,0,1,-1,1,1,2,2,1,
0,1,2,0,0,2,1,1,0,-1,1,0,2,1,
0,0,3,1,3,3,0,0,1,1,-1,2,1,3,
0,1,2,0,2,2,1,1,0,2,1,-1,2,1,
0,0,0,0,0,0,0,1,1,1,1,2,-1,0,
0,0,0,0,0,0,0,1,1,1,1,1,0,-1,
```

The example above is from a sample that is 14 columns by 14 rows, corresponding to the 
`int numNodes = 14` value in the NED file. In a topology with ‘n’ nodes, nodes are enumerated in the 
range [0, numNodes-1]. 

The first row represents `node0`, where the path to `node1` is via `port0`, to `node2` via
`port2`, to `node3` via `port1`, to `node4` via `port1` (so, by implication via `node3` 
which is also reached via `port1`), and so on.

In the 
[Network_nsfnetbw.ned](tests/unit-resources/nsfnetbw/Network_nsfnetbw.ned) file the
corresponding entries are:

```
node0.port[0] <--> Channel10kbps <--> node1.port[0];
node0.port[1] <--> Channel10kbps <--> node3.port[0];
node0.port[2] <--> Channel10kbps <--> node2.port[0];
```

Where we can see that the connection from `node0.port[0]` goes into `node1.port[0]`. In the
[Routing.txt](tests/unit-resources/nsfnetbw/Routing.txt) we also see that in the second row, for 
`node1`, the connection to `node0` is via `port0`.

### simulationResults.txt 

Thee files contain the samples generated by the OMNet++ network simulator. Each line in 
a 'simulationResults.txt' file corresponds to a simulation with different input traffic matrices, 
probabilistically generated from a given traffic intensity (\<lambda>). 

All of the samples are simulated using the routing scheme defined in 'Routing.txt'.

A given line of results is ~170K characters (including ','s), with the specific values at the 
positions in the results as described below.

The indices 'src_node' and 'dst_node' used below are in the range `[0, n-1]`, where `n` is the 
number of nodes in the topology, e.g. 14 for the nsfbetbw.

1. Bandwidth (in kbps) transmitted in each source-destination pair in the network 
(in both directions), where:

```  
bandwidth[src_node][dst_node] = (src_node∗n+dst_node)*3
```

2. Absolute number of packets transmitted in each source-destination pair (in both directions).

```  
pkts_transmitted[src_node][dst_node] = (src_node∗n+dst_node)*3 + 1
```
  
3. Absolute number of packets dropped in each source-destination pair.

```  
pkts_dropped[src_node][dst_node] = (src_node∗n+dst_node)*3 + 2
```  
   
4. Average per-packet delay over the packets transmitted in each source-destination pair. 

```  
avg_delay[src_node][dst_node] = n*n*3 + (src_node∗n+dst_node)*7  
```
  
5. Percentile 10 of the per-packet delay over the packets transmitted in each source-destination 
pair.

```  
delay_p_10[src_node][dst_node] = n*n*3 + (src_node∗n+dst_node)*7 + 1
```
  
6. Percentile 20 of the per-packet delay over the packets transmitted in each source-destination 
pair.
  
```  
delay_p_20[src_node][dst_node] = n*n*3 + (src_node∗n+dst_node)*7 + 2
```
  
7. Percentile 50 (median) of the per-packet delay over the packets transmitted in each 
source-destination pair.
 
```
delay_p_50[src_node][dst_node] = n*n*3 + (src_node∗n+dst_node)*7 + 3
```
  
8. Percentile 80 of the per-packet delay over the packets transmitted in each source-destination 
pair.

```  
delay_p_80[src_node][dst_node] = n*n*3 + (src_node∗n+dst_node)*7 + 4
```
  
9. Percentile 90 of the per-packet delay over the packets transmitted in each source-destination 
pair.
  
```  
delay_p_90[src_node][dst_node] = n*n*3 + (src_node∗n+dst_node)*7 + 5
```
  
10. Variance of the per-packet delay (jitter) over the packets transmitted in each 
source-destination pair.
  
```  
jitter[src_node][dst_node] = n*n*3 + (src_node∗n+dst_node)*7 + 6
```
    
### params.ini

Example contents:

```
[General]
**.simulationDuration = 16050
**.lambda = 9 
network = Network_synth50bw
repeat = 500
**.folderName = "data.5466564/synth50bw_9_Routing_SP_k_0_10_99"
``` 
 
Contains some input parameters used in our simulator. The most relevant ones at the user-level are 
'simulationDuration' and 'repeat'. 'simulationDuration' is the total duration of the simulation 
(in relative time units). For instance, this value is used to tune the simulation time in order to 
ensure that the behavior of performance metrics such as delay reaches a stationary state. Note that 
some simulation results like the total number of packets transmitted or dropped depend also on this 
parameter. Likewise, the 'repeat' parameter defines the number of simulation samples included in the 
current 'tar.gz' file.
    
* tfrecords/train: This directory contains the TFRecords files with the samples we used to train 
RouteNet in our [Demo paper](https://github.com/knowledgedefinednetworking/demo-routenet). To 
generate this training set, we randomly selected 80% of the samples in the original dataset.

* tfrecords/evaluate: This directory contains the TFRecords files used to evaluate RouteNet in our 
[Demo paper](https://github.com/knowledgedefinednetworking/demo-routenet). It contains a collection 
with the remaining 20% of the samples that were not present in the training set.