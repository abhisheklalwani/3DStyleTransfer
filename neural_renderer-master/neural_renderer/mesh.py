import torch
import torch.nn as nn

import neural_renderer as nr
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Mesh(object):
    '''
    A simple class for creating and manipulating trimesh objects
    '''
    def __init__(self, vertices, faces, textures=None, texture_size=4):
        '''
        vertices, faces and textures(if not None) are expected to be Tensor objects
        '''
        self.vertices = vertices
        self.faces = faces
        self.num_vertices = self.vertices.shape[0]
        self.num_faces = self.faces.shape[0]

        # create textures
        if textures is None:
            shape = (self.num_faces, texture_size, texture_size, texture_size, 3)
            self.textures = nn.Parameter(0.05*torch.randn(*shape))
            self.texture_size = texture_size
        else:
            self.texture_size = textures.shape[0]

    def get_batch(self,batch_size):
        vertices = torch.unsqueeze(self.vertices, 0)
        vertices = vertices.repeat(batch_size,1,1)
        faces = torch.unsqueeze(self.faces, 0)
        faces = faces.repeat(batch_size,1,1)
        textures = torch.unsqueeze(self.textures, 0)
        textures = torch.sigmoid(textures.repeat(batch_size,1,1,1,1,1))
        
        return vertices, faces, textures


    @classmethod
    def fromobj(cls, filename_obj, normalization=True, load_texture=False, texture_size=4):
        '''
        Create a Mesh object from a .obj file
        '''
        if load_texture:
            vertices, faces, textures = nr.load_obj(filename_obj,
                                                    normalization=normalization,
                                                    texture_size=texture_size,
                                                    load_texture=True)
        else:
            vertices, faces = nr.load_obj(filename_obj,
                                          normalization=normalization,
                                          texture_size=texture_size,
                                          load_texture=False)
            textures = None
        return cls(vertices, faces, textures, texture_size)
