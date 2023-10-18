import os
import os.path as osp
import shutil
from typing import Callable, List, Optional
import json
import numpy as np
import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import (
    coalesce,
    remove_self_loops,
    to_edge_index,
    to_torch_csr_tensor
)


class S2ORC(InMemoryDataset):
    r"""
    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'id_author.txt', 'id_paper.txt','id_venue.txt','id_topic.txt','S2ORC_gensci_Links_Author_Of.jsonl', 'S2ORC_gensci_Links_Published_In.jsonl',
            'S2ORC_gensci_Links_citations.jsonl', 'S2ORC_gensci_Links_topics.jsonl'
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass
    
    def read_json(self,path,cols=['from','to']):
        sample_data=[]
        with open(path) as f:
            for line in f:
                doc = json.loads(line)
                lst=[]
                for col in cols:
                    lst.append(doc[col])
                sample_data.append(lst)
        sample_data = pd.DataFrame(data=sample_data, columns=cols)
        return sample_data

    def process(self):
        import pandas as pd

        data = HeteroData()

        
        # Get paper labels.
        # master_path = '/rcfs/projects/expert/data/S2ORC_gensci/S2ORC_gensci_contextGraph_v2/S2ORC_full/raw/'
        master_path = '/shared/rcfs/expert/data/S2ORC_gensci/S2ORC_gensci_contextGraph_v2/S2ORC_full/raw/'
        ##path = osp.join(self.raw_dir, 'id_paper.txt')
        path = osp.join(master_path, 'id_paper.txt')
        paper = pd.read_csv(path, sep='\t', names=['name', 'idx'])
        print("#Papers: ",paper.shape)

        # Get author labels.
        ##path = osp.join(self.raw_dir, 'id_author.txt')
        path = osp.join(master_path, 'id_author.txt')
        author = pd.read_csv(path, sep='\t', names=['name', 'idx'])
        print("#Authors: ",author.shape)

        # Get venue labels.
        ##path = osp.join(self.raw_dir, 'id_venue.txt')
        path = osp.join(master_path, 'id_venue.txt')
        venue = pd.read_csv(path, sep='\t', names=['name', 'idx'])
        print("#Venues: ",venue.shape)

        # Get topic labels.
        ##path = osp.join(self.raw_dir, 'id_topic.txt')
        path = osp.join(master_path, 'id_topic.txt')
        topic = pd.read_csv(path, sep='\t', names=['name', 'idx'])
        print("#Topics: ",topic.shape)

        # Get paper<->author connectivity.
        path = osp.join(self.raw_dir, 'S2ORC_gensci_Links_Author_Of.jsonl')
        author_paper = self.read_json(path)
        author_paper=pd.merge(author_paper,author,left_on='from',right_on='name')
        author_paper=pd.merge(author_paper,paper,left_on='to',right_on='name')

        # Get paper<->venue connectivity.
        path = osp.join(self.raw_dir, 'S2ORC_gensci_Links_Published_In.jsonl')
        paper_venue = self.read_json(path)
        paper_venue=pd.merge(paper_venue,paper,left_on='from',right_on='name')
        paper_venue=pd.merge(paper_venue,venue,left_on='to',right_on='name')

        # Get paper<->paper connectivity
        path = osp.join(self.raw_dir, 'S2ORC_gensci_Links_citations.jsonl')
        paper_cites_paper = self.read_json(path)
        paper_cites_paper=pd.merge(paper_cites_paper,paper,left_on='from',right_on='name')
        paper_cites_paper=pd.merge(paper_cites_paper,paper,left_on='to',right_on='name')

        # Get paper<->topic connectivity.
        path = osp.join(self.raw_dir, 'S2ORC_gensci_Links_topics.jsonl')
        topic_paper = self.read_json(path)
        topic_paper=pd.merge(topic_paper,topic,left_on='from',right_on='name')
        topic_paper=pd.merge(topic_paper,paper,left_on='to',right_on='name')


        author_0=torch.from_numpy(author_paper['idx_x'].values)
        author_0_meta=torch.unique(author_0,return_counts=True)
        author_0=author_0_meta[0]

        venue_0=torch.from_numpy(paper_venue['idx_y'].values)
        venue_0_meta=torch.unique(venue_0,return_counts=True)
        venue_0=venue_0_meta[0]

        topic_0=torch.from_numpy(topic_paper['idx_x'].values)
        topic_0_meta=torch.unique(topic_0,return_counts=True)
        topic_0=topic_0_meta[0]

        paper_1=torch.from_numpy(author_paper['idx_y'].values)
        paper_2=torch.from_numpy(paper_venue['idx_x'].values)
        paper_3=torch.from_numpy(topic_paper['idx_y'].values)
        paper_0=torch.cat((paper_1,paper_2,paper_3),0)
        paper_0_meta=torch.unique(paper_0,return_counts=True)
        paper_0=paper_0_meta[0]
        

        data['paper'].y_index = torch.from_numpy(paper['idx'].values)#paper_0
        num_papers=paper.shape[0]#paper_0.size()[0]
        data['paper'].num_nodes = num_papers


        data['author'].y_index = torch.from_numpy(author['idx'].values)#author_0
        num_authors=author.shape[0]#author_0.size()[0]
        data['author'].num_nodes = num_authors


        data['venue'].y_index = torch.from_numpy(venue['idx'].values)#venue_0
        num_venues=venue.shape[0]#venue_0.size()[0]
        data['venue'].num_nodes = num_venues


        data['topic'].y_index = torch.from_numpy(topic['idx'].values)#topic_0
        num_topics=topic.shape[0]#topic_0.size()[0]
        data['topic'].num_nodes = num_topics



        author_paper = torch.from_numpy(author_paper[['idx_x','idx_y']].values)
        author_paper = author_paper.t().contiguous()
        author_paper = coalesce(author_paper, num_nodes=max(num_authors, num_papers))
        data['author', 'writes', 'paper'].edge_index = author_paper
        data['paper', 'written_by', 'author'].edge_index = author_paper.flip([0])    
        
#         adj = to_torch_csr_tensor(data['author', 'writes', 'paper'].edge_index, size=(M, N))
#         edge_index2, _ = to_edge_index(adj @ adj)
#         edge_index2, _ = remove_self_loops(edge_index2)
#         data['author', 'writes', 'paper'].edge_index = author_paper



        paper_venue = torch.from_numpy(paper_venue[['idx_x','idx_y']].values)
        paper_venue = paper_venue.t().contiguous()
        paper_venue = coalesce(paper_venue, num_nodes=max(num_papers, num_venues))
        data['paper', 'published_in', 'venue'].edge_index = paper_venue
        data['venue', 'publishes', 'paper'].edge_index = paper_venue.flip([0])
        


        paper_cites_paper = torch.from_numpy(paper_cites_paper[['idx_x','idx_y']].values)
        paper_cites_paper = paper_cites_paper.t().contiguous()
        paper_cites_paper = coalesce(paper_cites_paper, num_nodes=max(num_papers, num_papers))
        data['paper', 'outbound_cites', 'paper'].edge_index = paper_cites_paper
        data['paper', 'inbound_cites', 'paper'].edge_index = paper_cites_paper.flip([0])
        


        topic_paper = torch.from_numpy(topic_paper[['idx_x','idx_y']].values)
        topic_paper = topic_paper.t().contiguous()
        topic_paper = coalesce(topic_paper, num_nodes=max(num_topics, num_papers))
        
        data['topic', 'covered by', 'paper'].edge_index = topic_paper
        data['paper', 'covers', 'topic'].edge_index = topic_paper.flip([0])

        print(data)

        # subset_dict = {
        #         'paper': paper_0,#data['paper'].y_index,
        #         'author': author_0,#data['author'].y_index,
        #         'venue': venue_0,#data['venue'].y_index,
        #         'topic': topic_0,#data['topic'].y_index,
        #     }
        # data=data.subgraph(subset_dict)

        # print(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
        
    def __repr__(self) -> str:
        return 's2orc()'
        
#     def len(self):
#         return len(self.processed_file_names)

#     def get(self, idx):
#         data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
#         return data