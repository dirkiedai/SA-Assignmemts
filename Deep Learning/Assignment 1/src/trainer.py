import torch
import src.utils as utils
import logging
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, cfg, model, criterion, optimizer, train_set, valid_set):
        self.cfg = cfg
        self.cuda = torch.cuda.is_available() and not cfg.cpu
        self.device = torch.device("cuda") if self.cuda else torch.device("cpu")

        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer

        self.criterion = self.criterion.to(device=self.device)
        self.model = self.model.to(device=self.device)
        
        self.train_set = train_set
        self.valid_set = valid_set


    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.cfg.seed
        utils.set_torch_seed(seed)

    def train_step(self, sample, ignore_grad=False):
        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.optimizer.zero_grad()
        sample = utils.move_to_cuda(sample) if self.cuda else sample

        try:
            with torch.autograd.profiler.record_function("forward"):
                net_output = self.model(**sample["net_input"])
                loss = self.criterion(net_output, sample["target"])
            if ignore_grad:
                loss *= 0
            with torch.autograd.profiler.record_function("backward"):
        
                loss.backward()
                self.optimizer.step()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print('out of memory')
        return loss

    def valid_step(self, sample, raise_oom=False):
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            sample = utils.move_to_cuda(sample) if self.cuda else sample

            try:
                self.model.eval()
                with torch.no_grad():
                    net_output = self.model(**sample["net_input"])
                    loss = self.criterion(net_output, sample["target"])
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if not raise_oom:
                        logger.warning(
                            "ran out of memory in validation step, retrying batch"
                        )
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad = None  # free some memory
                        if self.cuda:
                            torch.cuda.empty_cache()
                        return self.valid_step(sample, raise_oom=True)     
                    raise e                  
            return loss

    def get_train_iterator(self):
        return DataLoader(self.train_set, batch_size=self.cfg.batch_size, collate_fn=self.train_set.collater, shuffle=True)

    def get_valid_iterator(self):
        return DataLoader(self.valid_set, batch_size=self.cfg.valid_batch_size, collate_fn=self.valid_set.collater, shuffle=False)







        


