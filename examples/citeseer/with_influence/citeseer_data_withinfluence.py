import dgl
import numpy as np
from pathlib import Path
import torch
from deepstochlog.term import Term, List
from deepstochlog.context import ContextualizedTerm, Context
from deepstochlog.dataset import ContextualizedTermDataset
from collections import defaultdict

root_path = Path(__file__).parent



dataset = dgl.data.CiteseerGraphDataset()
g = dataset[0]

# get node feature
documents = g.ndata['feat']
# get data split
train_ids = np.where(g.ndata['train_mask'].numpy())[0]
val_ids = np.where(g.ndata['val_mask'].numpy())[0]
test_ids = np.where(g.ndata['test_mask'].numpy())[0]

# get labels
labels = g.ndata['label'].numpy()

edges = []




citations = []
adjacency_list = defaultdict(list)
for eid in range(g.num_edges()):
    a, b = g.find_edges(eid)
    a, b = a.numpy().tolist()[0], b.numpy().tolist()[0],
    edges.append((a,b))
    citations.append("cite(%d, %d)." % (a,b))
    adjacency_list[a].append(b)
citations = "\n".join(citations)



class CiteseerDataset(ContextualizedTermDataset):
    def __init__(
        self,
        split: str,
        labels,
        documents):
        if split == "train":
            self.ids = train_ids
        elif split =="valid":
            self.ids = val_ids
        elif split == "test":
            self.ids = test_ids
        else:
            raise Exception("Unkonw split %s" % split)
        self.labels = labels
        self.is_test = True if split in ("test", "valid") else False
        self.documents = documents
        self.dataset = []

        context = {List(Term(str(i))): (d,) for i, d in enumerate(self.documents)}
        for i in range(6):
            context[List(Term("class" + str(i)))] = (torch.tensor([i]),)

        # for h in adjacency_list:
        #     all = adjacency_list[h]
        #     l = List(*[Term(str(i)) for i in all])
        #     for j, t in enumerate(all):
        #         context[List(Term(str(h)), Term(str(t)), l)] = (self.documents[h], torch.tensor([j]), self.documents[all])
        for h in adjacency_list:
            all = adjacency_list[h]
            l = List(*[Term(str(i)) for i in all])
            for j, t in enumerate(all):
                context[List(Term(str(h)), Term(str(t)), l)] = (
                torch.tensor([h]), torch.tensor([j]), torch.tensor(all))

        context = Context(context)
        self.queries_for_model = []
        for did in self.ids:
            label = List("class" + str(self.labels[did]))
            query = ContextualizedTerm(
                context=context,
                term=Term("s", did, label))
            self.dataset.append(query)
            if self.is_test:
                query_model = Term("s", did, List("_"))
            else:
                query_model = query.term
            self.queries_for_model.append(query_model)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if type(item) is slice:
            return (self[i] for i in range(*item.indices(len(self))))
        return self.dataset[item]



train_dataset = CiteseerDataset(split="train", documents=documents, labels=labels)
valid_dataset = CiteseerDataset(split="valid", documents=documents, labels=labels)
test_dataset = CiteseerDataset(split="test", documents=documents, labels=labels)

queries_for_model = train_dataset.queries_for_model + valid_dataset.queries_for_model + test_dataset.queries_for_model





