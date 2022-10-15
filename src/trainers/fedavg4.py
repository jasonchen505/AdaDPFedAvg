from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import LrdWorker
from src.optimizers.gd import GD
import numpy as np
import math
import torch

criterion = torch.nn.CrossEntropyLoss()


class FedAvg4Trainer(BaseTrainer):
    """
    Scheme I and Scheme II, based on the flag of self.simple_average
    """

    def __init__(self, options, dataset):
        self.data_num = list()

        self.model = choose_model(options)
        self.move_model_to_gpu(self.model, options)

        self.optimizer = GD(self.model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        self.num_epoch = options['num_epoch']  # E
        worker = LrdWorker(self.model, self.optimizer, options)
        super(FedAvg4Trainer, self).__init__(options, dataset, worker=worker)

        self.selected_times = [0 for i in range(self.clients_num)]

        self.prob = self.compute_prob()

        # privacy loss related parameters
        self.delta = 1e-5
        self.sigma = 0.9
        self.l_max = 32
        self.epsilon = 0
        self.w_clip = 60
        self.km = [[0 for i in range(self.l_max + 1)] for j in range(self.clients_num)]

    def local_train(self, round_i, selected_clients, **kwargs):
        """Training procedure for selected local clients

        Args:
            round_i: i-th round training
            selected_clients: list of selected clients

        Returns:
            solns: local solutions, list of the tuple (num_sample, local_solution)
            stats: Dict of some statistics
        """
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs
        for i, c in enumerate(selected_clients, start=1):
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)

            # Solve minimization locally
            soln, stat = c.local_train()

            # Compute the LDP privacy loss for each client
            pl = self.compute_privacy_loss_advanced(self.selected_times[c.cid], c.cid)
            self.epsilon = max(self.epsilon, pl)
            if self.print_result:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Privacy Loss: {:>.5f} | "
                      "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                    round_i, c.cid, i, self.clients_per_round,
                    pl,  # the accumulated privacy loss for the client
                    stat['norm'], stat['min'], stat['max'],
                    stat['loss'], stat['acc'] * 100, stat['time']))

            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)

        return solns, stats

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()

        # initialization for pl
        self.init_coefficient()

        for round_i in range(self.num_round):
            # Test latest model on train data
            self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)

            # Choose K clients prop to data size
            selected_clients, repeated_times = self.select_clients_with_prob(seed=round_i)

            # Solve minimization locally
            solns, stats = self.local_train(round_i, selected_clients)
            print(self.epsilon)

            # Track communication cost
            self.metrics.extend_commu_stats(round_i, stats)

            # Update latest model
            self.latest_model = self.aggregate(solns, repeated_times=repeated_times)
            self.optimizer.inverse_prop_decay_learning_rate(round_i)

        # Test final model on train data
        self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)

        # Save tracked information
        # 存到json文件中，以人类可读的方式
        self.metrics.write()

    def compute_prob(self):
        probs = []
        for c in self.clients:
            probs.append(len(c.train_data))
        return np.array(probs) / sum(probs)

    def select_clients_with_prob(self, seed=1):
        num_clients = min(self.clients_per_round, len(self.clients))
        np.random.seed(seed)
        index = np.random.choice(len(self.clients), num_clients, p=self.prob)
        index = sorted(index.tolist())

        select_clients = []
        select_index = []
        repeated_times = []
        for i in index:
            if i not in select_index:
                select_clients.append(self.clients[i])
                select_index.append(i)
                repeated_times.append(1)
            else:
                repeated_times[-1] += 1

        # to compute the accumulated PL
        for i in index:
            self.selected_times[i] += 1

        return select_clients, repeated_times

    def aggregate(self, solns, **kwargs):
        averaged_solution = torch.zeros_like(self.latest_model)
        # averaged_solution = np.zeros(self.latest_model.shape)
        if self.simple_average:
            repeated_times = kwargs['repeated_times']
            assert len(solns) == len(repeated_times)
            for i, (num_sample, local_solution) in enumerate(solns):
                # repeated_times[i] maybe means one client can be selected multiple times by uniform randomization
                # the noise addition position should be moved into client to be closer to the actual situation
                local_solution = local_solution / max(1, torch.norm(local_solution) / self.w_clip)
                sensitivity=torch.norm(local_solution)
                averaged_solution += (local_solution +
                                      (self.sigma ** 2) * (sensitivity ** 2) *
                                      torch.randn(local_solution.shape)) * repeated_times[i]
            averaged_solution /= self.clients_per_round
        else:
            for num_sample, local_solution in solns:
                averaged_solution += num_sample * local_solution
            averaged_solution /= self.all_train_data_num
            averaged_solution *= (100 / self.clients_per_round)
        # print(averaged_solution)
        print(torch.norm(averaged_solution))
        return averaged_solution.detach()

    def log_moment_generating_func(self, l, q):
        m = 0
        for k in range(0, l + 1):
            # calculate C(alpha, k)
            comb_num = math.factorial(l) / (math.factorial(k) * math.factorial(l - k))
            # notice: the exp() will easily cause overflow!! so sigma under 1 is dangerous
            m += comb_num * (1 - q) ** (l - k) * q ** k * math.exp((k ** 2 - k) / (2 * self.sigma ** 2))
        return math.log(m)

    def init_coefficient(self):
        for i in range(self.clients_num):
            q = self.batch_size / self.data_num[i]
            print(i, q)
            for j in range(1, self.l_max + 1):
                self.km[i][j] = self.log_moment_generating_func(j, q)

    def compute_privacy_loss_advanced(self, step, client):
        e = 1e9
        for l in range(1, self.l_max + 1):
            epsilon_with_l = (self.km[client][l] * step + math.log(1 / self.delta)) / l
            # print(l, epsilon_with_l)
            e = min(epsilon_with_l, e)
        return e
