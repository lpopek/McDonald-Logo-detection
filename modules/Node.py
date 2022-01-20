class Node:
    def __init__(self, row, col, col_cls):
        self.row = row
        self.col = col
        self.cls = col_cls
    
    def __str__(self):
        return f"Row: {self.row} Column = {self.col} Color = {self.cls}" 

    def __eq__(self, other):
        return ((self.row, self.col, self.cls) == (other.row, other.col, other.cls))