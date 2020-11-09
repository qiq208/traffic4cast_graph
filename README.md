# traffic4cast_graph
Team Road Rage - Graph Neural Network  - Pak Hay Kwok (pak_hay_kwok@hotmail.com) and Qi Qi (qiq208@gmail.com)

This is the graph neural network submission for Team Road Rage

## To run the Berlin Submission
There are quite a few pre-processing steps, most of the small data files for Berlin have been uploaded to save having to re-run

From notebooks:

Run `create_graph_dataset.ipynb` - this will process all the raw images into graph data and store the data in 3 .h5 files (training, validation, testing)

Run `train.ipynb` - this will do the training and the submission


## To run on other cities 
Prior to running the above notebooks, you will need to run two python scripts (and change the city names) in the src folder:

Run `gnn_make_static_grid.py` - this calculates the overall max volume for each pixel across all channels

Run `gnn_make_nodes_edges.py` - this will create the nodes and edges information for the road network

