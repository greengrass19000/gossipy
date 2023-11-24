import torch
from torch.nn.modules.loss import CrossEntropyLoss
from networkx import to_numpy_array
from networkx.generators.random_graphs import random_regular_graph
from gossipy import set_seed
from gossipy.core import UniformDelay, AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork
from gossipy.node import GossipNode, PartitioningBasedNode, SamplingBasedNode
from gossipy.model.handler import PartitionedTMH, SamplingTMH, TorchModelHandler
from gossipy.model.sampling import TorchModelPartition
from gossipy.model.nn import LogisticRegression
from gossipy.data import load_classification_dataset, DataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.simul import SimulationReport, TokenizedGossipSimulator
from gossipy.flow_control import RandomizedTokenAccount
from gossipy.utils import plot_evaluation


set_seed(123)
X, y = load_classification_dataset("spambase", as_tensor=True)
data_handler = ClassificationDataHandler(X, y, test_size=.1)
dispatcher = DataDispatcher(data_handler, n=100, eval_on_user=False, auto_assign=True)
# topology = StaticP2PNetwork(100, to_numpy_array(random_regular_graph(20, 100, seed=42)))
topology = StaticP2PNetwork(100, topology=None)
net = LogisticRegression(data_handler.Xtr.shape[1], 2)

nodes = PartitioningBasedNode.generate(
    data_dispatcher=dispatcher,
    p2p_net=topology,
    round_len=100,
    model_proto=PartitionedTMH(
        net=net,
        tm_partition=TorchModelPartition(net, 4),
        optimizer=torch.optim.SGD,
        optimizer_params={
            "lr": 1,
            "weight_decay": .001
        },
        criterion=CrossEntropyLoss(),
        create_model_mode=CreateModelMode.UPDATE #CreateModelMode.MERGE_UPDATE
    ),
    sync=True
)

simulator = TokenizedGossipSimulator(
    nodes=nodes,
    data_dispatcher=dispatcher,
    token_account=RandomizedTokenAccount(C=20, A=10),
    utility_fun=lambda mh1, mh2, msg: 1,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH, 
    delay=UniformDelay(0, 10),
    sampling_eval=.1
)

report = SimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)
simulator.start(n_rounds=1000)

plot_evaluation([[ev for _, ev in report.get_evaluation(False)]], "Overall test results")