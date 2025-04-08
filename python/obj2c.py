import sys
import os
import re

name = sys.argv[1]

class Surface:
    def __init__(self):
        self.colour = (255,255,255)
        self.indices = []
        
vertices = []
edges = []
surfaces = []
surface = Surface()
surfaces.append(surface)

with open(name, 'r') as obj_file:
    for line in obj_file:
        tokens = line.split()
        #print(f'Processing: 1')
        if tokens[0] == 'v':
            vertices.append((tokens[1], tokens[2], tokens[3]))
        elif tokens[0] == 'f':
            zero = -1
            prev = int(tokens[-1].split('/')[0])-1
            for vtn in tokens[1:]:
                curr = int(vtn.split('/')[0])-1
                if zero < 0:
                    zero = curr
                elif prev != zero:
                    surface.indices.append(zero)
                    surface.indices.append(prev)
                    surface.indices.append(curr)
                prev = curr
                
        elif tokens[0] == 'l':
            prev = int(tokens[-1].split('/')[0])-1
            for vtn in tokens[1:]:
                curr = int(vtn.split('/')[0])-1
                edge = (min(prev, curr), max(prev, curr), surface.colour)
                if edge not in edges:
                    edges.append(edge)
                prev = curr
                    
        elif tokens[0] == 'usemtl':
            surface = Surface()
            surfaces.append(surface)
            
            
if (len(vertices) >= 3):
    identifier = re.sub(r'[^a-zA-Z0-9_]', '_', os.path.splitext(os.path.basename(name))[0])
    
    surfaces = [surface for surface in surfaces if surface.indices]
    
    print( '#define E_ RGBPIX(255,255,255)')
    print(f'static const model_t model_{identifier} = {{')
    print(f'    .vertex_count = {len(vertices)},')
    print( '    .vertices = (vertex_t*)(float[][5]){', end='')
    for v, vert in enumerate(vertices):
        if ((v % 8) == 0):
            print('\n        ', end='')
        print(f'{{{float(vert[0]):g}, {float(vert[2]):g}, {float(vert[1]):g}}}, ', end='')
    print( '\n    },')
    
    print(f'    .surface_count = {len(surfaces)},')
    print( '    .surfaces = (surface_t[]){')
    for surface in surfaces:
        print(f'        {{{len(surface.indices)}, (index_t[]){{', end='')
        for i, idx in enumerate(surface.indices):
            if ((i % 40) == 0):
                print('\n            ', end='')
            print(f'{idx}, ', end='')
        print(f'\n        }}, RGBPIX{surface.colour}}},')
    print('    },')
    
    print(f'    .edge_count = {len(edges)},')
    print( '    .edges = (edge_t[]){', end='')
    for e, edge in enumerate(edges):
        if ((e % 16) == 0):
            print('\n        ', end='')
        print(f'{{{{{edge[0]}, {edge[1]}}}, E_}},', end='')
    print( '\n    }')
    print( '};')
    print("#undef E_")