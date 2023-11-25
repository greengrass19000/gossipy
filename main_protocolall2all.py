import torch
from torch.nn.modules.loss import CrossEntropyLoss
from gossipy import set_seed
from gossipy.core import AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork, UniformMixing
from gossipy.node import All2AllGossipNode
from gossipy.model.handler import WeightedTMH
from gossipy.model.nn import LogisticRegression
from gossipy.data import load_classification_dataset, DataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.simul import All2AllGossipSimulator, SimulationReport
from gossipy.utils import plot_evaluation

seed = 42
set_seed(seed)
X, y = load_classification_dataset("spambase", as_tensor=True)
data_handler = ClassificationDataHandler(X, y, test_size=.1)
dispatcher = DataDispatcher(data_handler, n=100, eval_on_user=False, auto_assign=True)
topology = StaticP2PNetwork(dispatcher.size(), topology=None)
net = LogisticRegression(data_handler.Xtr.shape[1], 2)

nodes = All2AllGossipNode.generate(
    data_dispatcher=dispatcher,
    p2p_net=topology,
    model_proto=WeightedTMH(
        net=net,
        optimizer=torch.optim.SGD,
        optimizer_params={
            "lr": .1,
            "weight_decay": .01
        },
        criterion=CrossEntropyLoss(),
        create_model_mode=CreateModelMode.MERGE_UPDATE),
    round_len=100,
    sync=False
)

simulator = All2AllGossipSimulator(
    nodes=nodes,
    data_dispatcher=dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    sampling_eval=.1
)

report = SimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=seed)
simulator.start(UniformMixing(topology), n_rounds=20)

plot_evaluation([[ev for _, ev in report.get_evaluation(local=False)]], "Overall test results")
