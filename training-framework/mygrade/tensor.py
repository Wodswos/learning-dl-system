import math


# 四则运算 + 指数运算应该肯定是要实现的
# 在此基础上应该就能实现大部分初等函数的 forward/backward
# 话说回来，四则这种基础运算和初等函数的关系是什么？？
class Scalar():
    def __init__(self, value, prev=(), grad=0) -> None:
        self.value = value
        self.grad = grad

        # self._prev 是计算图的前驱节点——当然是对于前向计算过程而言的
        self._prev = prev

    def __repr__(self) -> str:
        return f'Scalar(value={self.value})'

    def convert(self, input) -> "Scalar":
        if isinstance(input, Scalar):
            return input

        # convert int/float to value.
        try:
            scalar_instance = Scalar(input)
        except Exception:
            raise RuntimeError(f"Object Type {type(input)} can't convert to Scalar.")

        return scalar_instance

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        other = self.convert(other)

        output = Scalar(self.value + other.value, prev=(self, other))

        # 这里是直接把函数赋值给 output 的属性，所以定义时不需要 self 参数
        def _backward() -> None:
            # 很容易想到的代码实现是： self.grad= output.grad; other.grad = output.grad
            # If self & other 是两个独立变量，这很 OK. But:
            # 1. If other == self，此时实际上是 output = 2 * self，那么 set 两次 self.grad = output.grad 显然是不对的。
            # 2. If other != self，但 other 由 self 计算得到（比如 other = self ** 2），
            #    （假设 output.grad=1）此时手算一下正确的 self.grad 应为 (self.value+1)，而不是 1 或者self.value （透过 other 传播）

            # 也就是说对一个因变量分开求了两次导数 —— 那么该怎么解决呢？backward 的时候不是直接 set 而是 add？

            self.grad = self.grad + output.grad
            other.grad = other.grad + output.grad

        output._backward = _backward

        return output

    def __radd__(self, other):
        # 加法 & 乘法满足交换律，可以直接这么写
        # 减法 & 除法就需要 convert other 变量并调用 other.__xxx__(self) 了
        return self.__add__(other)

    def __sub__(self, other):
        other = self.convert(other)

        output = Scalar(
            self.value - other.value,
            prev=(self, other,)
        )

        def _backward() -> None:
            self.grad = self.grad + output.grad
            other.grad = other.grad - output.grad

        output._backward = _backward

        return output

    def __rsub__(self, other):
        other = self.convert(other)
        return other.__sub__(self)

    def __mul__(self, other):
        other = self.convert(other)

        output = Scalar(
            self.value * other.value,
            prev=(self, other,)
        )

        def _backward() -> None:
            self.grad = self.grad + other.value * output.grad
            other.grad = other.grad + self.value * output.grad

        output._backward = _backward

        return output

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # 也可以直接调 self * other ** (-1)，这里还是单独实现了一遍
        other = self.convert(other)

        output = Scalar(
            self.value / other.value,
            prev=(self, other,)
        )

        def _backward() -> None:
            self.grad = self.grad + output.grad * (1 / other.value)
            other.grad = other.grad + output.grad * (- self.value / (other.value ** 2))

        output._backward = _backward

        return output

    def __rtruediv__(self, other):
        other = self.convert(other)
        return other.__truediv__(self)

    def __pow__(self, exponent):
        exponent = self.convert(exponent)

        output = Scalar(
            value=self.value ** exponent.value,
            prev=(self, exponent,)
        )

        def _backward() -> None:
            # 应用幂函数求导规则
            self.grad = self.grad + output.grad * (exponent.value * self.value ** (exponent.value - 1))
            # 应用指数函数求导规则
            exponent.grad = exponent.grad + output.grad * (self.value ** exponent.value * math.log(self.value))

        output._backward = _backward

        return output

    def __rpow__(self, base):
        base = self.convert(base)
        return base.__pow__(self)

    def tanh(self):
        e = Scalar(math.e)
        output = (1 - e ** (-2 * self)) / (1 + e ** (-2 * self))

        return output

    def _backward(self) -> None:
        # default backward, do nothing
        pass

    def backward(self) -> None:
        sorted_queue = []
        visited_node = set()

        def _topo_sort(node: Scalar) -> None:
            # 1. 一般来说，拓扑排序的实现是依次把入度（或者出度）为 0 的节点添加到列表中
            # 2. 在 Scalar 的实现中，每个节点记录且仅记录了 self.prev（从反向传播过程视角来说是后继节点），即出度信息很好获取
            #
            # 所以此处拓扑排序的实现，是将出度为 0 的节点逐步添加到队列
            for successor in node._prev:
                if successor not in visited_node:
                    _topo_sort(successor)

            sorted_queue.append(node)
            visited_node.add(node)

        _topo_sort(self)
        # 需要注意，由于是依据出度得来的拓扑序，所以需要反转一下
        sorted_queue.reverse()
        # print(sorted_queue)

        # 反向传播
        self.grad = 1
        for node in sorted_queue:
            # print(node)
            node._backward()


def tanh(input: Scalar) -> Scalar:
    # 下面这种写法的代价是会有十几个中间变量——但我不 care 性能
    # 可以手写 tanh forward 和 backward，把大量的中间变量去掉
    output = (1 - math.e ** (-2 * input)) / (1 + math.e ** (-2 * input))

    return output


if __name__ == '__main__':
    # 验证代码不拆分，视为 Test case，每次执行全跑一遍，怎么不算一种 CI 呢 😄

    # test basic arithmetic operation
    a = Scalar(4)
    b = Scalar(3)

    print(f'a + b = {a + b}')
    print(f'a * b = {a * b}')

    # test topological sort, 暂时还没实现 reset grad
    # simple test 1
    a = Scalar(3.0)
    b = a + a
    b.backward()
    assert a.grad == 2

    # simple test 2
    a = Scalar(3.0)
    b = Scalar(4.0)
    c = a * b
    d = c + c  # 还没实现 int * scalar，暂时用加法
    d.backward()
    assert a.grad == 8
    assert c.grad == 2
    assert b.grad == 6

    # operation between Scalar and Python data type (int/float).
    a = Scalar(3.0)
    b = Scalar(4.0)
    c = a * b
    d = c * 2
    d.backward()

    assert c.grad == 2
    assert a.grad == 8
    assert b.grad == 6

    # test divide

    # test tanh function
    logits = Scalar(100)
    activation = tanh(logits)
    activation.backward()
    print(f'tanh({logits}) = {activation}, grad is {logits.grad}')

    logits = Scalar(0)
    activation = tanh(logits)
    activation.backward()
    print(f'tanh({logits}) = {activation}, grad is {logits.grad}')

    logits = Scalar(-10)
    activation = tanh(logits)
    activation.backward()
    print(f'tanh({logits}) = {activation}, grad is {logits.grad}')

    # from utils import draw_dot
    # dot = draw_dot(activation)
    # dot.render('calculate_tanh', view=False)
