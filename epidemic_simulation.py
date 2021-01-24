import numpy as np
import networkx as nx
import random

'''
    @Author: Itay Dar
    @Date: 2021-01-14 12:54:04
'''


class Epidemic_Simulation:
    ''' Creating an epidemic simulation with given parameters,
        In order to detect the epidemic behaviour and find a path to get over it.

        Given Parameters
        -----------
        @ network:
            The network to run the analysis over, a networkX.Graph object.
            The network contains edge information (infection probability) as well as node information (the status of the node - ‘s’-suspected, ‘i’-infected or ‘r’-recoverd).
        @ seed:
            The seed to be used for simulation
        @ model_type:
            The epidemic model behavior. Can be either ‘SIS’ or ‘SIR’
        @ p:
            Probability of an infectious node to pass the disease to another node he is in contact with (in a single meeting). float 0-1.
        @ infection_time:
            Number of epochs that an infectious person “carries” the disease and risks his susceptible neighbors.
        @ epoches:
            Number of epochs (e.g., days) to apply the simulation over

        Global Parameters
        -----------------
        @ infected:
            dictionary contain all the infected nodes as keys, and how much time they carry the infectection as value. (e.g: {node : 2})
        @ edges_weights:
            dictionary contain all the edges as keys, and the number of meeting per epoch between the edges nodes. (e.g: {(u,v)) : 8})
        @ status:
            dictionary contain all the nodes as keys and their infection status and mortality likelihood as value. (e.g: {u : {'mortality_likelihood: 0.31',status: 'i'}}
        @ neighbors:
            dictionary contain all the nodes as keys and list of their neighbors as value (e.g:{u : [v,x,z,d]})
        @ infections_total:
            variable that count the number of infection during the simulation
        @ mortality_total:
            variable that count the number of death during the simulation
    '''

    def __init__(self, network, seed, model_type, p, infection_time, epochs):
        self.network = network.copy(as_view=False)
        self.model_type = model_type
        self.p = p
        self.infection_time = infection_time
        self.epochs = epochs

        self.infected = self.infected()
        self.edges_weight = self.create_weight_dic()
        self.status = dict(self.network.nodes(data=True))
        self.neighbors = self.create_edge_dict_non_dir()

        self.infections_total = 0
        self.mortality_total = 0
        np.random.seed(seed=seed)

    def create_edge_dict_non_dir(self):
        """
        @Name : create_edge_dict_non_dir
        @Do: for a given network, create dictionary that for each node will store list of his neighbors.
        @ Param:
                no parameters.
        @ Return:
                dict
        """

        edges_dict = {}
        for edge in self.network.edges():
            if edge[0] in edges_dict:
                edges_dict[edge[0]].append(edge[1])
            else:
                edges_dict[edge[0]] = [edge[1]]

            if edge[1] in edges_dict:
                edges_dict[edge[1]].append(edge[0])
            else:
                edges_dict[edge[1]] = [edge[0]]

        return edges_dict

    def create_weight_dic(self):
        """
        @Name : create_edge_dict_non_dir
        @Do: for a given network, create dictionary that for each node will store list of his neighbors.
        @ Param:
                no parameters.
        @ Return:
                dict
        """

        dic = {}
        for info in self.network.edges(data=True):
            dic[(info[0], info[1])] = info[2]['contacts']
        return dic

    def infected(self):
        """
        @Name : infected
        @Do: initialized dict that for each infected node in the beginning of the simulation
            will store 0 as the number of epochs the node is infected.
        @ Param:
                no parameters.
        @ Return:
                dict
        """

        infected_dic = {}
        temp = nx.get_node_attributes(self.network, 'status')
        for node in temp:
            if temp[node] == 'i':
                infected_dic[node] = 0  # set initial infected data structure
        return infected_dic

    def check_if_dead(self):
        """
        @Name : check_if_dead
        @Do: for each infected node in infected dict, draw a random number between [0,1].
            check if the random number is smaller then the node mortality likelihood.
            if so, "kill" the node, update the mortality_total, the node status and the infected dic. otherwise, the node survived another epoc -> update his infected[node]
        @ Param:
                no extrernal parameters.
        @ Return:
                dict
        """
        infected_list = [i for i in list(self.infected.keys()) if self.infected[i] != 0]  # new infected nodes cannot die in the epoch they infected.
        thread_of_life = np.random.rand(1, len(infected_list))[0] # set random array that draw random number between [0,1] - death probability.
        if len(infected_list) > 0:
            for i in range(len(thread_of_life)):
                if thread_of_life[i] < self.status[infected_list[i]]['mortalitylikelihood']:
                    self.mortality_total += 1
                    self.status[infected_list[i]] ['status'] = 'd'
                    del self.infected[infected_list[i]]
        return

    def infect_step(self):
        """
        @Name : infect_step
        @Do: for each infected node chaeck -> for each of his healthy neighbors do -> draw a random number between [0,1].
            if the random number is smaller than the probability p to infect with the given times of meetings - then pass the virus to the uninfected node and update neccecery fields.
            otherwise, the neighbor has not been infected.
        @ Param:
                None - just change global parameters
        @ Return:
                None - just change global parameters
        """

        for infected_node in list(self.infected.keys()):
            self.infected[infected_node] += 1
            temp_neighbors = [i for i in self.neighbors[infected_node] if self.status[i]['status'] == 's']
            infected_array = np.random.rand(1, len(temp_neighbors))[0]
            for i in range(len(temp_neighbors)):
                cur_neighbor = temp_neighbors[i]
                e = (infected_node, cur_neighbor) if (infected_node, cur_neighbor) in self.edges_weight else (
                cur_neighbor, infected_node)
                contacts = self.edges_weight[e]
                if infected_array[i] < 1 - ((1 - self.p) ** contacts):
                    self.status[cur_neighbor]['status'] = 'i'
                    self.infected[cur_neighbor] = 0
                    self.infections_total += 1
        return

    def manage_healthy(self):
        """
        @Name : manage_healthy
        @Do:  -> if node exceeded the number of infection time, send him free and change his status according to the epidimiological model.
        @ Param:
                None - just change global parameters
        @ Return:
                None - just change global parameters
        """

        for node in list(self.infected.keys()):
            if self.infected[node] == self.infection_time:
                del self.infected[node]
                if self.model_type == "SIS":
                    self.status[node]['status'] = 's'
                else:
                    self.status[node]['status'] = 'r'
        return


    def calculate_r_0(self):
        #in order to calculate r_0 i will use this equation: (number of nodes predicted to infect/number of current infected nodes)

        r_0 = 0

        for infected_node in self.infected.keys():
            temp_neighbors = [i for i in self.neighbors[infected_node] if self.status[i]['status'] == 's']
            for neighbor in temp_neighbors:
                e = (infected_node, neighbor) if (infected_node, neighbor) in self.edges_weight else (
                    neighbor, infected_node)
                contacts = self.edges_weight[e]
                infected_probability = 1 - ((1-self.p)**contacts)
                r_0 += infected_probability
        try:
            return r_0/len(self.infected)
        except ZeroDivisionError:
            return 0

    def vaccination(self, policy, vaccines):
        """
        @Name : vaccination
        @Do:  -> according to the policy and the number of vaccines avaliable, change the chosen nodes from i/s to r.
                basically choose the best nodes from the network that fit the parameter.
        @ Param:
                Vaccintation - str - given nodes sorting by parameter. such as: highest betweness,
                                                                                random selection,
                                                                                degree centrality,
                                                                                 mortality likelihood.
        @ Return:
                None - just change global parameters
        """

        if policy == 'rand':
            lucky_nodes = random.choices(list(self.network.nodes()), k=vaccines)
        elif policy == 'betweenness':
            lucky_nodes = sorted([(k, v) for k, v in nx.betweenness_centrality(self.network).items()],
                                 key=lambda x: x[1], reverse=True)[:vaccines]
            lucky_nodes = [node[0] for node in lucky_nodes]
        elif policy == 'degree':
            lucky_nodes = sorted([(k, v) for k, v in nx.degree_centrality(self.network).items()], key=lambda x: x[1],
                                 reverse=True)[:vaccines]
            lucky_nodes = [node[0] for node in lucky_nodes]
        else:
            lucky_nodes = sorted([(k, v['mortalitylikelihood']) for k, v in self.status.items()], key=lambda x: x[1],
                                 reverse=True)[:vaccines]
            lucky_nodes = [i[0] for i in lucky_nodes]
        for node in lucky_nodes:
            self.status[node]['status'] = 'r'
            if node in list(self.infected.keys()):
                del self.infected[node]
        return


def epidmeic_analysis(network, model_type='SIS', infection_time=2, p=0.05, epochs=20, seed=312541915):
    """
    @Name : epidmeic_analysis
    @Do:  -> the main function. create Epidemic_Simulation instance.
            for number of epochs do -> manage healthy nodes -> do the inection step -> check if there are deads -> repeat.
            extract the relevant data from the outcomes and return them as dict.

    @ Param:
            network - nx.Graph object with weigthed edges and status, mortality likelihood for each node.
            model_type - the epidimiologic model that our simulation is behaving.
            infectio_time: how many epoches the node contain the virus.
            p: the probabilty to pass the epidemic from one node to another per meeting.
            epoches - the number of time units which the epidemic lasts
            seed - to get better statistical results. use in np.random(seed=seed).
    @ Return:
            dictionary: {infected total: int
                        infectioius_current: int
                        mortality_total: in
                        r_0: int}
    """

    epi = Epidemic_Simulation(network, seed, model_type, p, infection_time, epochs=epochs)
    steps = 0
    while steps < epochs:
        epi.manage_healthy()
        epi.infect_step()
        epi.check_if_dead()
        steps += 1
    epi.manage_healthy()
    infections_total = epi.infections_total
    inf_cur = len(epi.infected)
    mor_total = epi.mortality_total
    return {'infections_total': infections_total,
            'infectioius_current': inf_cur,
            'mortality_total': mor_total,
            'r_0': epi.calculate_r_0()}


def vaccination_analysis(network, model_type="SIR", infection_time=2, p=0.05, epochs=10, seed=312541915, vaccines=1,
                         policy='rand'):
    """
    @Name : vaccination_analysis
    @Do:    the main function. create Epidemic_Simulation instance.
            first give vaccine as number as vaccines avalible to the best candidate across the network
            for number of epochs do -> manage healthy nodes -> do the inection step -> check if there are deads -> repeat.
            extract the relevant data from the outcomes and return them as dict.
    @ Param:
            network - nx.Graph object with weigthed edges and status, mortality likelihood for each node.
            model_type - the epidimiologic model that our simulation is behaving.
            infectio_time: how many epoches the node contain the virus.
            p: the probabilty to pass the epidemic from one node to another per meeting.
            epoches - the number of time units which the epidemic lasts
            seed - to get better statistical results. use in np.random(seed=seed).
            vaccines - number of avalivble vaccines to use.
            policy - how to choose who will get vaccine.
    @ Return:
            dictionary: {infected total: int
                        infectioius_current: int
                        mortality_total: in
                        r_0: int}
    """

    epi = Epidemic_Simulation(network, seed, model_type, p, infection_time, epochs=epochs)
    epi.vaccination(policy,
                    vaccines)  # give vaccines and cure nodes. once nodes got vaccine, he is no longer in threat.
    steps = 0
    while steps < epochs:
        epi.manage_healthy()
        epi.infect_step()
        epi.check_if_dead()
        steps += 1
    epi.manage_healthy()
    infections_total = epi.infections_total
    inf_cur = len(epi.infected)
    mor_total = epi.mortality_total
    return {'infections_total': infections_total,
            'infectioius_current': inf_cur,
            'mortality_total': mor_total,
            'r_0': epi.calculate_r_0()}
