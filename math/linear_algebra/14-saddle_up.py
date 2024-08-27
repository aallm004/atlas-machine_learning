#!/usr/bin/env python3
def matrix_shape(matrix):
  if not matrix:
    return []
  
  shape = []
  current_level = matrix

  while isinstance(current_level, list):
    shape.append(len(current_level))
    first_row_length = len(current_level[0]) if current_level else 0
    for row in current_level:
      if len(row) != first_row_length:
        raise ValueError("Matrix is not rectangular")
      current_level = current_level[0]

  return shape
