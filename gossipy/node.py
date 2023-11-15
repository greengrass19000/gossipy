from __future__ import annotations
import random
import numpy as np
from numpy.random import randint, normal, rand
from numpy import ndarray
from torch import Tensor
from typing import Any, Optional, Union, Dict, Tuple, Iterable
from gossipy.data import DataDispatcher

from . import CACHE, LOG
from .core import AntiEntropyProtocol, CreateModelMode, MessageType, Message, P2PNetwork, ChordMessage
from .utils import choice_not_n
from .model.handler import ModelHandler, PartitionedTMH, SamplingTMH, WeightedTMH
from .model.sampling import TorchModelSampling

from math import log2

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "Apache License, Version 2.0"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ["GossipNode",
           "PassThroughNode",
           "CacheNeighNode",
           "SamplingBasedNode",
           "PartitioningBasedNode",
           "PENSNode",
           "ChordNode"]


class GossipNode():
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 model_handler: ModelHandler, #object that handles the model learning/inference
                 p2p_net: P2PNetwork,
                 sync: bool=True):
        r"""Class that represents a generic node in a gossip network. 

        A node is identified by its index and it is initialized with a fixed delay :math:`\Delta` that
        represents the idle time. The node can be either synchronous or asynchronous. In the former case,
        the node will time out exactly :math:`\Delta` time steps into the round. Thus it is assumed that 
        :math:`0 < \Delta <` `round_len`. In the latter case, the node will time out to every 
        :math:`\Delta` time steps. In the synchronous case, :math:`\Delta \sim U(0, R)`, otherwise 
        :math:`\Delta \sim \mathcal{N}(R, R/10)` where :math:`R` is the round length.

        Parameters
        ----------
        idx : int
            The node's index.
        data : tuple[Tensor, Optional[Tensor]] or tuple[ndarray, Optional[ndarray]]
            The node's data in the format :math:`(X_\text{train}, y_\text{train}), (X_\text{test}, y_\text{test})`
            where :math:`y_\text{train}` and :math:`y_\text{test}` can be `None` in the case of unsupervised learning.
            Similarly, :math:`X_\text{test}` and :math:`y_\text{test}` can be `None` in the case the node does not have
            a test set. 
        round_len : int
            The number of time units in a round.
        model_handler : ModelHandler
            The object that handles the model learning/inference.
        p2p_net: P2PNetwork
            The peer-to-peer network that provides the list of reachable nodes according to the 
            network topology.
        sync : bool, default=True
            Whether the node is synchronous with the round's length. In this case, the node will 
            regularly time out at the same point in the round. If `False`, the node will time out 
            with a fixed delay. 
        """

        self.idx: int = idx
        self.data:  Union[Tuple[Tensor, Optional[Tensor]], Tuple[ndarray, Optional[ndarray]]] = data
        self.round_len: int = round_len
        self.model_handler: ModelHandler = model_handler
        self.sync: bool = sync
        self.delta: int = randint(0, round_len) if sync else int(normal(round_len, round_len/10))
        self.p2p_net = p2p_net

    def init_model(self, local_train: bool=True, *args, **kwargs) -> None:
        """Initializes the local model.

        Parameters
        ----------
        local_train : bool, default=True
            Whether the local model should be trained for with the local data after the
            initialization.
        """

        self.model_handler.init()
        if local_train:
            self.model_handler._update(self.data[0])

    def get_peer(self) -> int:
        """Picks a random peer from the reachable nodes.

        Returns
        -------
        int
            The index of the randomly selected peer.
        """

        peers = self.p2p_net.get_peers(self.idx)
        return random.choice(peers) if peers else choice_not_n(0, self.p2p_net.size(), self.idx)
        
    def timed_out(self, t: int) -> bool:
        """Checks whether the node has timed out.
        
        Parameters
        ----------
        t : int
            The current timestamp.
        
        Returns
        -------
        bool
            Whether the node has timed out.
        """
        return ((t % self.round_len) == self.delta) if self.sync else ((t % self.delta) == 0)
    

    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Message:
        """Sends a message to the specified peer.

        The method actually prepares the message that will be sent to the peer.
        The sending is performed by the simluator and it may be delayed or it can fail.

        Parameters
        ----------
        t : int
            The current timestamp.
        peer : int
            The index of the peer node.
        protocol : AntiEntropyProtocol
            The protocol used to send the message.

        Returns
        -------
        Message
            The message to send.
        
        Raises
        ------
        ValueError
            If the protocol is not supported.
        
        See Also
        --------
        :class:`gossipy.simul.GossipSimulator`
        """

        if protocol == AntiEntropyProtocol.PUSH:
            key = self.model_handler.caching(self.idx)
            return Message(t, self.idx, peer, MessageType.PUSH, (key,))
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            key = self.model_handler.caching(self.idx)
            return Message(t, self.idx, peer, MessageType.PUSH_PULL, (key,))
        else:
            raise ValueError("Unknown protocol %s." %protocol)

    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        """Receives a message from the peer.

        After the message is received, the local model is updated and merged according to the `mode`
        of the model handler. In case of a pull/push-pull message, the local model is sent back to the
        peer.

        Parameters
        ----------
        t : int
            The current timestamp.
        msg : Message
            The received message.
        
        Returns
        -------
        Message or `None`
            The message to be sent back to the peer. If `None`, there is no message to be sent back.
        """

        msg_type: MessageType
        recv_model: Any 
        msg_type, recv_model = msg.type, msg.value[0] if msg.value else None
        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            recv_model = CACHE.pop(recv_model)
            self.model_handler(recv_model, self.data[0])

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            key = self.model_handler.caching(self.idx)
            return Message(t, self.idx, msg.sender, MessageType.REPLY, (key,))
        return None

    def evaluate(self, ext_data: Optional[Any]=None) -> Dict[str, float]:
        """Evaluates the local model.

        Parameters
        ----------
        ext_data : Any, default=None
            The data to be used for evaluation. If `None`, the local test data will be used.
        
        Returns
        -------
        dict[str, float]
            The evaluation results. The keys are the names of the metrics and the values are
            the corresponding values.
        """

        if ext_data is None:
            return self.model_handler.evaluate(self.data[1])
        else:
            return self.model_handler.evaluate(ext_data)
    
    #CHECK: we need a more sensible check
    def has_test(self) -> bool:
        """Checks whether the node has a test set.

        Returns
        -------
        bool
            Whether the node has a test set.
        """

        if isinstance(self.data, tuple):
            return self.data[1] is not None
        else: return True
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__} #{self.idx} (Δ={self.delta})"


    @classmethod
    def generate(cls,
                 data_dispatcher: DataDispatcher,
                 p2p_net: P2PNetwork,
                 model_proto: ModelHandler,
                 round_len: int,
                 sync: bool,
                 **kwargs) -> Dict[int, GossipNode]:
        """Generates a set of nodes.

        Parameters
        ----------
        data_dispatcher : DataDispatcher
            The data dispatcher used to distribute the data among the nodes.
        p2p_net : P2PNetwork
            The peer-to-peer network topology.
        model_proto : ModelHandler
            The model handler prototype.
        round_len : int
            The length of a round in time units.
        sync : bool
            Whether the nodes are synchronized with the round length or not.

        Returns
        -------
        Dict[int, GossipNode]
            The generated nodes.
        """
        
        nodes = {}
        for idx in range(p2p_net.size()):
            node = cls(idx=idx,
                       data=data_dispatcher[idx], 
                       round_len=round_len, 
                       model_handler=model_proto.copy(), 
                       p2p_net=p2p_net, 
                       sync=sync, 
                       **kwargs)
            nodes[idx] = node
        return nodes    
#############################################################################################################  
# Nam Dung 2023
class ChordNode(GossipNode):
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 model_handler: WeightedTMH, #object that handles the model learning/inference
                 p2p_net: P2PNetwork,
                 sync: bool=True):
        super(ChordNode, self).__init__(idx,
                                        data,
                                        round_len,
                                        model_handler,
                                        p2p_net,
                                        sync)
        m = int(log2(p2p_net.size())) + 1
        self.finger = [1]*m
        pow2 = 1
        cur = idx
        # print('another')
        # print(self.idx)
        for i in range(m):
            cur = idx + pow2
            pow2 = pow2 + pow2
            if cur >= p2p_net.size():
                cur = cur - p2p_net.size()
            self.finger[i] = cur
        #     print(cur, end=' ')
        # print('')
        self.finger.reverse()
        self.local_cache = {}
    
    # docstr-coverage:inherited
    def timed_out(self, t: int, weights: Iterable[float]) -> int:
        tout = super().timed_out(t)
        if tout and self.local_cache:
            #TODO: don't delete the model imediately, remain it log() time steps
            self.model_handler([CACHE.pop(k) for k in self.local_cache.values()], self.data[0], weights)
            self.local_cache = {}
        return tout 
    
    def next_timed_out(self, t: int) -> int:
        tmp = t % self.delta
        tmp = self.delta - tmp
        return tmp
    
    def get_peers(self) -> int:
        return self.finger

    # docstr-coverage:inherited
    def send(self,
             t: int,
             sender: int,
             peer: int,
             protocol: AntiEntropyProtocol,
             limit: int) -> Union[ChordMessage, None]:

        if protocol == AntiEntropyProtocol.PUSH:
            key = self.model_handler.caching(sender)
            return ChordMessage(t, sender, peer, limit, MessageType.PUSH, (key,))
        else:
            raise ValueError("ChordNode only supports PUSH protocol.")

    # docstr-coverage:inherited
    def receive(self, t: int, msg: ChordMessage) -> Union[ChordMessage, None]:
        msg_type: MessageType
        recv_model: Any 
        sender, msg_type, recv_model = msg.sender, msg.type, msg.value[0] if msg.value else None
        if msg_type == MessageType.PUSH:
            # this should never happen
            if sender in self.local_cache:
                # LOG.info("something happened")
                #TODO: Chờ đến lúc đến thằng limit rồi hẵng xóa
                if self.idx == msg.limit:        
                    CACHE.pop(self.local_cache[sender])
            self.local_cache[sender] = recv_model
            # return msg
        return None
#############################################################################################################

# Hegedus 2021
class SamplingBasedNode(GossipNode):
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 model_handler: SamplingTMH, #object that handles the model learning/inference
                 p2p_net: P2PNetwork,
                 sync=True):
        super(SamplingBasedNode, self).__init__(idx,
                                                data,
                                                round_len,
                                                model_handler,
                                                p2p_net,
                                                sync)

    # docstr-coverage:inherited          
    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Union[Message, None]:

        if protocol == AntiEntropyProtocol.PUSH:
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH,
                           (key, self.model_handler.sample_size))
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH_PULL,
                           (key, self.model_handler.sample_size))
        else:
            raise ValueError("Unknown protocol %s." %protocol)

    # docstr-coverage:inherited
    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: Any 
        msg_type = msg.type

        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            recv_model, sample_size = msg.value
            recv_model = CACHE.pop(recv_model)
            sample = TorchModelSampling.sample(sample_size, recv_model.model)
            self.model_handler(recv_model, self.data[0], sample)

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           msg.sender,
                           MessageType.REPLY,
                           (key, self.model_handler.sample_size))
        return None

# Hegedus 2021
class PartitioningBasedNode(GossipNode):
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 model_handler: PartitionedTMH, #object that handles the model learning/inference
                 p2p_net: P2PNetwork,
                 sync=True):
        r"""Standard :class:`GossipNode` with partitioned model.

        This type of node has been first introduced in :cite:p:`Hegedus:2021`.
        The only difference with the standard :class:`GossipNode` is that the model stored
        in the node is partitioned. Thus, both the :meth:`send` and :meth:`receive` methods
        handle the partitioning.

        Parameters
        ----------
        idx : int
            The node's index.
        data : tuple[Tensor, Optional[Tensor]] or tuple[ndarray, Optional[ndarray]]
            The node's data in the format :math:`(X_\text{train}, y_\text{train}), (X_\text{test}, y_\text{test})`
            where :math:`y_\text{train}` and :math:`y_\text{test}` can be `None` in the case of unsupervised learning.
            Similarly, :math:`X_\text{test}` and :math:`y_\text{test}` can be `None` in the case the node does not have
            a test set. 
        round_len : int
            The number of time units in a round.
        model_handler : PartitionedTMH
            The object that handles the learning/inference of partitioned-based models.
        p2p_net: P2PNetwork
            The peer-to-peer network that provides the list of reachable nodes according to the 
            network topology.
        sync : bool, default=True
            Whether the node is synchronous with the round's length. In this case, the node will 
            regularly time out at the same point in the round. If `False`, the node will time out 
            with a fixed delay. 
        """
        super(PartitioningBasedNode, self).__init__(idx,
                                                    data,
                                                    round_len,
                                                    model_handler,
                                                    p2p_net,
                                                    sync)

    # docstr-coverage:inherited            
    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Union[Message, None]:

        if protocol == AntiEntropyProtocol.PUSH:
            pid = np.random.randint(0, self.model_handler.tm_partition.n_parts)
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH,
                           (key, pid))
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            pid = np.random.randint(0, self.model_handler.tm_partition.n_parts)
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH_PULL,
                           (key, pid))
        else:
            raise ValueError("Unknown protocol %s." %protocol)

    # docstr-coverage:inherited
    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: Any 
        msg_type = msg.type

        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            recv_model, pid = msg.value
            recv_model = CACHE.pop(recv_model)
            self.model_handler(recv_model, self.data[0], pid)

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            pid = np.random.randint(0, self.model_handler.tm_partition.n_parts)
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           msg.sender,
                           MessageType.REPLY,
                           (key, pid))
        return None

# Koloskova et al. 2020
class All2AllGossipNode(GossipNode):
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 model_handler: WeightedTMH, #object that handles the model learning/inference
                 p2p_net: P2PNetwork,
                 sync: bool=True):
        r"""
        TODO

        Parameters
        ----------
        idx : int
            The node's index.
        data : tuple[Tensor, Optional[Tensor]] or tuple[ndarray, Optional[ndarray]]
            The node's data in the format :math:`(X_\text{train}, y_\text{train}), (X_\text{test}, y_\text{test})`
            where :math:`y_\text{train}` and :math:`y_\text{test}` can be `None` in the case of unsupervised learning.
            Similarly, :math:`X_\text{test}` and :math:`y_\text{test}` can be `None` in the case the node does not have
            a test set. 
        round_len : int
            The number of time units in a round.
        model_handler : ModelHandler
            The object that handles the model learning/inference.
        p2p_net: P2PNetwork
            The peer-to-peer network that provides the list of reachable nodes according to the 
            network topology.
        sync : bool, default=True
            Whether the node is synchronous with the round's length. In this case, the node will 
            regularly time out at the same point in the round. If `False`, the node will time out 
            with a fixed delay. 
        use_mh : bool, default=False
            Whether to use the Metropolis-Hastings weighting scheme for the model averaging.
        """
        super(All2AllGossipNode, self).__init__(idx,
                                        data,
                                        round_len,
                                        model_handler,
                                        p2p_net,
                                        sync)
        self.local_cache = {}
    
    # docstr-coverage:inherited
    def timed_out(self, t: int, weights: Iterable[float]) -> int:
        tout = super().timed_out(t)
        if tout and self.local_cache:
            # if not self.use_mh:
            #     weights = [1.0 /(1 + len(self.local_cache))] * (len(self.local_cache) + 1)
            # else:
            #     n = self.p2p_net.size(self.idx)
            #     weights = [1./n] + [1. / (min(self.p2p_net.size(k), n) + 1) for k in self.local_cache]
            self.model_handler([CACHE.pop(k) for k in self.local_cache.values()], self.data[0], weights)
            self.local_cache = {}
        return tout 

    def get_peers(self) -> int:
        return self.p2p_net.get_peers(self.idx)

    # docstr-coverage:inherited
    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Union[Message, None]:

        if protocol == AntiEntropyProtocol.PUSH:
            return super().send(t, peer, protocol)
        else:
            raise ValueError("All2AllNode only supports PUSH protocol.")

    # docstr-coverage:inherited
    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: Any 
        sender, msg_type, recv_model = msg.sender, msg.type, msg.value[0] if msg.value else None
        if msg_type == MessageType.PUSH:
            # this should never happen
            if sender in self.local_cache:
                CACHE.pop(self.local_cache[sender])
            self.local_cache[sender] = recv_model

        return None