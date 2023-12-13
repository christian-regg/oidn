import sys
import os.path
import numpy as np
from onnx import *
from onnx.parser import *

class OpDict:
    def __init__(self):
        self.ops = dict()

    def add_node(self, name) -> None:
        if self.ops.get(name):
            value = self.ops[name]
        else:
            value = 0

        self.ops[name] = value + 1

    def merge(self, other_dict):
        for name, value in sorted(other_dict.ops.items()):
            if self.ops.get(name):
                newValue = self.ops[name] + value
            else:
                newValue = 1

            self.ops[name] = newValue

    def print(self, sorted_by_value = False) -> None:
        if sorted_by_value:
            for name in sorted(self.ops, key=self.ops.get, reverse=True):
                print("   %s:%d" % (name, self.ops[name]))
        else:
            for name, count in sorted(self.ops.items()):
                print("   %s:%d" % (name, count))

    

def model_info(model_path) -> dict():
    print("Visiting:%s" % model_path)    
    model = onnx.load(model_path)

    print("   ir_version:", model.ir_version)
    for opset in model.opset_import:
        print("   opset domain=%r version=%r" % (opset.domain, opset.version))

    ops = OpDict()
    for n in model.graph.node:
        ops.add_node(n.op_type)

    ops.print()

    return ops

def walk_tree(root, files) -> None:
    for entry in os.listdir(root):
        path = os.path.join(root, entry)
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1]
            if ext.lower() == ".onnx":
                files.append(path)
        else:
            walk_tree(path, files)

    pass

if __name__ == "__main__":

    files = []

    if os.path.isdir(sys.argv[1]):
        walk_tree(sys.argv[1], files)
    else:
        files.append(sys.argv[1])

    all_ops = OpDict()
    for f in files:
        model_ops = model_info(f)
        all_ops.merge(model_ops)

    print("Total")
    all_ops.print(True)

        

            