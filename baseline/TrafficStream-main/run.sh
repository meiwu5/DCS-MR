#!/bin/bash

#PEMSD3-stream
# python main.py --conf config/PEMSD3-stream/expansible.json --dataset PEMSD3-stream --gpuid 1
# python main.py --conf config/PEMSD3-stream/retrained.json --dataset PEMSD3-stream --gpuid 1
# python main.py --conf config/PEMSD3-stream/static.json --dataset PEMSD3-stream --gpuid 1
# python main.py --conf config/PEMSD3-stream/trafficStream.json --dataset PEMSD3-stream --gpuid 1

#PEMSD4-stream
# python main.py --conf config/PEMSD4-stream/expansible.json --dataset PEMSD4-stream --gpuid 1
# python main.py --conf config/PEMSD4-stream/retrained.json --dataset PEMSD4-stream --gpuid 1
# python main.py --conf config/PEMSD4-stream/static.json --dataset PEMSD4-stream --gpuid 1
python main.py --conf config/PEMSD4-stream/trafficStream.json --dataset PEMSD4-stream --gpuid 1

#PEMSD8-stream
# python main.py --conf config/PEMSD8-stream/expansible.json --dataset PEMSD8-stream --gpuid 1
# python main.py --conf config/PEMSD8-stream/retrained.json --dataset PEMSD8-stream --gpuid 1
# python main.py --conf config/PEMSD8-stream/static.json --dataset PEMSD8-stream --gpuid 1
# python main.py --conf config/PEMSD8-stream/trafficStream.json --dataset PEMSD8-stream --gpuid 1