chord_notes = {'major7': ['c4', 'e4', 'g4', 'b4'],
          'minor7': ['c4', 'ef4', 'g4', 'bf4'],
          'majorminor7': ['c4', 'e4', 'g4', 'bf4'],
          'minormajor7': ['c4', 'ef4', 'g4', 'b4'],
          'halfdim7': ['c4', 'ef4', 'gf4', 'bf4'],
          'fulldim7': ['c4', 'ef4', 'gf4', 'a4'],
          'augmajor7': ['c4', 'e4', 'gs4', 'b4'],
          'augminor7': ['c4', 'e4', 'gs4', 'bf4'],
          }

num_chords = len(chord_notes.keys())
quality_to_idx = {x:i for (i,x) in enumerate(chord_notes.keys())}
idx_to_quality = {i:x for (x,i) in quality_to_idx.items()}
class_arr = [idx_to_quality[i] for i in range(num_chords)]
