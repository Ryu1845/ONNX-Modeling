import numpy as np
import onnx
import os


class Model:
    def __init__(
        self,
        name: str,
        ops_version: int = 15,
        externalData=True,
        *args,
        dtype=None,
        **kwargs,
    ):
        self.dtype = (
            onnx.TensorProto.FLOAT
            if dtype == np.float32
            else onnx.TensorProto.FLOAT16
            if dtype == np.float16
            else onnx.TensorProto.BFLOAT16
            if dtype == np.bfloat16
            else onnx.TensorProto.FLOAT
        )
        self.nptype = (
            np.float32
            if dtype == onnx.TensorProto.FLOAT
            else np.float16
            if dtype == onnx.TensorProto.FLOAT16
            else np.float16
            if dtype == onnx.TensorProto.BFLOAT16
            else np.float32
        )

        self.nm = 0
        exportname = f"{name}_{'32' if dtype == onnx.TensorProto.FLOAT else '16'}_{ops_version}.onnx"
        externalname = f"{name}_{'32' if dtype == onnx.TensorProto.FLOAT else '16'}_{ops_version}.bin"

        # remove old files
        if os.path.exists(exportname):
            os.remove(exportname)
        if os.path.exists(externalname):
            os.remove(externalname)

        self.ops_version = ops_version
        self.TensorList = []
        self.NodeList = []
        #self.one = initTensor([1.0] * embed)
        #self.margins = initTensor([0.00001] * embed)
        self.module = object

        # pytorch function defs
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x
        self.mainfunc = lambda x: x
        self.stackEmbed = False

        self.emptyState = np.array([], dtype=self.nptype)
        # self.zero = initTensor([0.0]*embed)

    def init_tensor(self, x):
        name = f"PreTrainedTensor_{self.nm}"
        self.nm += 1
        if isinstance(x, list):
            xx = np.array(x).astype(self.nptype)
        else:
            xx = x.squeeze().float().cpu().numpy()
            # convert to float32
            xx = xx.astype(self.nptype)
        rrx = onnx.helper.make_tensor(
            name, self.dtype, xx.shape, xx.tobytes(), raw=True
        )

        if externalData:
            onnx.external_data_helper.set_external_data(
                rrx,
                location=externalname,
            )

        self.TensorList.append(rrx)
        return name

    def sqrt(self, x):
        name = f"sqrt_{self.nm}_out"
        self.nm += 1
        node = onnx.helper.make_node("Sqrt", inputs=[x], outputs=[name])
        self.NodeList.append(node)

        return name

    def mean(self, x):
        name = f"mean_{self.nm}_out"
        self.nm += 1
        node = onnx.helper.make_node("ReduceMean", inputs=[x], outputs=[name])
        self.NodeList.append(node)

        return name

    def relu(self, x):
        name = f"relu_{self.nm}_out"
        self.nm += 1
        node = onnx.helper.make_node("Relu", inputs=[x], outputs=[name])
        self.NodeList.append(node)

        return name

    def exp(self, x):
        name = f"exp_{self.nm}_out"
        self.nm += 1
        node = onnx.helper.make_node("Exp", inputs=[x], outputs=[name])
        self.NodeList.append(node)

        return name

    def stack(self, x):
        return [initTensor(r) for r in x]

    def matvec(self, x, y):
        name = f"matvec_{self.nm}_out"
        self.nm += 1
        node = onnx.helper.make_node("MatMul", inputs=[x, y], outputs=[name])
        self.NodeList.append(node)
        return name

    def prod(self, x):
        name = f"prod_{self.nm}_out"
        self.nm += 1
        node = onnx.helper.make_node(
            "ReduceProd", inputs=[x], outputs=[name], axes=[1], keepdims=0
        )
        self.NodeList.append(node)

        return name

    def mul(self, x, y):
        name = f"mul_{self.nm}_out"
        self.nm += 1
        node = onnx.helper.make_node("Mul", inputs=[x, y], outputs=[name])
        self.NodeList.append(node)

        return name

    def squeeze(self, x):
        name = f"squeeze_{self.nm}_out"
        self.nm += 1
        node = onnx.helper.make_node("Squeeze", inputs=[x], outputs=[name])
        self.NodeList.append(node)

        return name

    def add(self, x, y):
        name = f"add_{self.nm}_out"
        self.nm += 1
        node = onnx.helper.make_node("Add", inputs=[x, y], outputs=[name])
        self.NodeList.append(node)

        return name

    def sub(self, x, y):
        name = f"sub_{self.nm}_out"
        self.nm += 1
        node = onnx.helper.make_node("Sub", inputs=[x, y], outputs=[name])
        self.NodeList.append(node)

        return name

    def lerp(self, x, y, z):
        return self.add(x, self.multiply(self.subtract(y, x), z))

    def minimum(self, x, y):
        name = f"minimum_{self.nm}_out"
        self.nm += 1
        node = onnx.helper.make_node("Min", inputs=[x, y], outputs=[name])
        self.NodeList.append(node)

        return name

    def log(self, x):
        name = f"log_{self.nm}_out"
        self.nm += 1
        node = onnx.helper.make_node("Log", inputs=[x], outputs=[name])
        self.NodeList.append(node)

        return name

    def divide(self, x, y):
        name = f"divide_{self.nm}_out"
        self.nm += 1
        node = onnx.helper.make_node("Div", inputs=[x, y], outputs=[name])
        self.NodeList.append(node)

        return name

    # ort 15 does not support layernorm
    def layernorm(self, x, w, b):
        def layernorm17(x, w, b):
            name = f"layernorm_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                "LayerNormalization", inputs=[x, w, b], outputs=[name]
            )
            self.NodeList.append(node)

            return name

        if self.ops_version >= 17:
            return layernorm17(x, w, b)
        xee2 = self.subtract(x, self.mean(x))
        x2 = self.add(
            self.sqrt(self.add(self.mean(self.multiply(xee2, xee2)), self.margins)),
            self.margins,
        )
        return self.add(self.multiply(w, self.divide(xee2, x2)), b)

    def get_index(self, x, y):
        name = f"getIndex_{self.nm}_out"
        self.nm += 1
        node = onnx.helper.make_node("Gather", inputs=[x, y], outputs=[name])
        self.NodeList.append(node)

        return squeeze(name)

    def neg(self, x):
        name = f"neg_{self.nm}_out"
        self.nm += 1
        node = onnx.helper.make_node("Neg", inputs=[x], outputs=[name])
        self.NodeList.append(node)

        return name

    def logistical(self, x):
        name = f"logistic_{self.nm}_out"
        self.nm += 1
        node = onnx.helper.make_node("Sigmoid", inputs=[x], outputs=[name])
        self.NodeList.append(node)

        return name

    def maximum(self, x, y):
        name = f"maximum_{self.nm}_out"
        self.nm += 1
        node = onnx.helper.make_node("Max", inputs=[x, y], outputs=[name])
        self.NodeList.append(node)

        return name

    def ppm(self, x, layers, embed, useSafeWKV=True):
        inputtensor = (
            onnx.helper.make_tensor_value_info("input0", onnx.TensorProto.INT32, [1]),
            "input0",
        )

        emptyState = list(
            map(
                lambda x: (
                    onnx.helper.make_tensor_value_info(
                        "instate" + str(x), self.dtype, [embed]
                    ),
                    "instate" + str(x),
                ),
                range((4 + useSafeWKV) * layers),
            )
        )
        outs = x.forward(inputtensor[1], list(map(lambda x: x[1], emptyState)))
        print(self.TensorList.__len__())
        print(self.NodeList.__len__())
        print(outs)
        logits = onnx.helper.make_tensor_value_info(outs[0], self.dtype, [50277])
        state = list(
            map(
                lambda x: onnx.helper.make_tensor_value_info(x, self.dtype, [embed]),
                outs[1],
            )
        )

        # Create the graph (GraphProto)
        graph_def = onnx.helper.make_graph(
            nodes=self.NodeList,  # The list of nodes in the graph.
            name="RWKV",
            # Graph input
            inputs=[inputtensor[0], *list(map(lambda x: x[0], emptyState))],
            outputs=[logits, *state],  # Graph output
            initializer=self.TensorList,  # initializer
            # did not work, needs to be external
        )

        modelDef = onnx.helper.make_model(
            graph_def,
            producer_name="rwkvstic",
        )

        modelDef.opset_import[0].version = opsVersion

        onnx.save(modelDef, exportname)

        # run model
        print("Model saved to: ", exportname, " and is ready to be run")
        print("Data type: ", self.dtype)
        print("Embedding size: ", embed)
        print("Number of layers: ", layers)
        print("external data: ", externalname)
