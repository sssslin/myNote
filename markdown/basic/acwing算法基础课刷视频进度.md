# acwing算法基础课刷视频进度

前情提要：**算法需要熟记算法模板，相对的，数据结构，需要记住其基本增删改查操作，这个是非常重要的，如果不记住数据结构的基本实现方式，则不可能做出相应的题目。**



## 基础算法3节

第一节核心内容

排序：快排、归并排序

二分：整数二分、浮点数二分

上课：主要思想理解    下课：在理解的基础上，背模板    下课：acwing配套习题，一道题目写3-5遍，增加熟练度



### 快排

基于分治思想

1. 确定分界点q[l]、q[(l + r) / 2]、q[r]、随机
2. 调整范围，左边的数字 <= x,右边的数 >= x  (难点，如何简单优雅的处理)
3. 递归处理左右两段

```java
// 快速排序算法模板
void quick_sort(int q[], int l, int r)
{
    if (l >= r) return;
    
    int i = l - 1, j = r + 1, x = q[l];
    while (i < j)
    {
        do i ++ ; while (q[i] < x);
        do j -- ; while (q[j] > x);
        if (i < j) swap(q[i], q[j]);
        else break;
    }
    quick_sort(q, l, j), quick_sort(q, j + 1, r);
}
```



优雅的进行快排的方法：双指针分别从左边界和右边界相遇而行



### 归并排序

time:O（nlogn）

算法思想：找分界点

1. 确定分界点， mid = (l + r) / 2

2. 递归排序left、right

3. 归并,合二为一（难点）

   ```java
   // 归并排序算法模板
   void merge_sort(int q[], int l, int r)
   {
       if (l >= r) return;
       
       int mid = l + r >> 1;
       merge_sort(q, l, mid);
       merge_sort(q, mid + 1, r);
       
       int k = 0, i = l, j = mid + 1;
       while (i <= mid && j <= r)
           if (q[i] < q[j]) tmp[k ++ ] = q[i ++ ];
           else tmp[k ++ ] = q[j ++ ];
       
       while (i <= mid) tmp[k ++ ] = q[i ++ ];
       while (j <= r) tmp[k ++ ] = q[j ++ ];
       
       for (i = l, j = 0; i <= r; i ++, j ++ ) q[i] = tmp[j];
   }
   ```

   稳定排序的概念



### 整数二分

有单调性一定能二分，没有单调性也可能可以进行二分

O(logn )



## 基础算法第二节

### 高精度（加减乘除）

**注：Java中由于BigInteger存在所以没有相关内容，优先级可以稍微降低一点点**

当前先贴上代码，后续再仔细补充

```java
// 高精度加法
// C = A + B, A >= 0, B >= 0
vector<int> add(vector<int> &A, vector<int> &B)
{
    if (A.size() < B.size()) return add(B, A);
    
    vector<int> C;
    int t = 0;
    for (int i = 0; i < A.size(); i ++ )
    {
        t += A[i];
        if (i < B.size()) t += B[i];
        C.push_back(t % 10);
        t /= 10;
    }
    
    if (t) C.push_back(t);
    return C;
}	
```

```java
// 高精度减法
// C = A - B, 满足A >= B, A >= 0, B >= 0
vector<int> sub(vector<int> &A, vector<int> &B)
{
    vector<int> C;
    for (int i = 0, t = 0; i < A.size(); i ++ )
    {
        t = A[i] - t;
        if (i < B.size()) t -= B[i];
        C.push_back((t + 10) % 10);
        if (t < 0) t = 1;
        else t = 0;
    }

    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
```

```java
// 高精度乘低精度
// C = A * b, A >= 0, b > 0
vector<int> mul(vector<int> &A, int b)
{
    vector<int> C;
    int t = 0;
    for (int i = 0; i < A.size() || t; i ++ )
    {
        if (i < A.size()) t += A[i] * b;
        C.push_back(t % 10);
        t /= 10;
    }
    
    return C;
}
```

```java
// 高精度除以低精度
// A / b = C ... r, A >= 0, b > 0
vector<int> div(vector<int> &A, int b, int &r)
{
    vector<int> C;
    r = 0;
    for (int i = A.size() - 1; i >= 0; i -- )
    {
        r = r * 10 + A[i];
        C.push_back(r / b);
        r %= b;
    }
    reverse(C.begin(), C.end());
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
```





### 前缀和(一维)

前缀和：计算这个元素及其前面元素的和

前缀和与差分的作用都是将区间运算化为点运算，降低时间复杂度

这是一种数组预处理的方式，主要的作用，或者唯一的租用就是一定区间内的和。



计算a数组区间[m, n]的区间和，可以表示为

设sum[]表示一个区间的和，则

m与n之间的区间和可以认为是[0,m],[0,n]之间的差

sum[m - 1] =  a<sub>1</sub> + a<sub>2</sub> + a<sub>3</sub>+ .... a<sub>m-1</sub>

sum[n] = a<sub>1</sub> + a<sub>2</sub> + a<sub>3</sub>+ .... a<sub>m-1</sub> + a<sub>m</sub> + a<sub>m +1</sub> .... a<sub>n</sub>两者一减即可得到以下表达式

**a[m]+a[m+1]+…+a[n] = sum[n] - sum[m-1]** 



在实际应用中，一般求区间问题，需要先构建一个前缀和数组

```java
 		// 由于 nums 全都是正整数，因此 preSum 严格单调增加
        // preSum 表示 sum(nums[0..i))
        int[] nums = {};
        int len = nums.length;
        int[] preSum = new int[len + 1];
        preSum[0] = 0;
        for (int i = 0; i < len; i++) {
            preSum[i + 1] = preSum[i] + nums[i];
        }
```



### 前缀和（二维）

二维前缀和，指的是在二维平面上计算指定区域内的点的元素之和，二维前缀和依然可以化区间计算为点子算，将（mn）的时间复杂度化为（1）。

![](D:\DownloadAndData\private\Java\myNote\markdown\basic\二维前缀和图示.png)

绿框和棕框都包含sum[m-1] [n-1]这个部分，所以相当于加了两次，所以要剪掉一次，最后再补上a[m] [n]即可

sum[m] [n] = a[m] [n] + sum[m-1] [n] + sum[m] [n-1] - sum[m-1] [n-1]



# 差分数组（一维）

概念：将数列中的每一项分别与前一项数做差

差分数组的作用：对一个数组区间内的元素进行加减，达到O(1)的复杂度



性质：

- 差分序列求前缀和可得原序列

- 将原序列区间[L,R]中的元素全部+1，可以转化操作为差分序列L处+1，R+1处-1

- 按照性质2得到，每次修改原序列一个区间+1，那么每次差分序列修改处增加的和减少的相同

  

  性质二说明

  a【l】= d【1】+ d 【2】+ …+d【l】,当d[l] + c，a[l]以及a[l]之后由于包含d[l],所以会自动加c，为了避免后续所有数据都加C，所以要在R+1处-c,让R之后的数据回归正常 （具体讲解见，acwing基础算法2,1:50:00处）




## 基础算法第三节

### 双指针

(0-1:00:00小时)

做双指针的题目的时候，首先想一下，暴力的做法是怎么样的，然后再去分析其中的单调性规律，通过这个规律去优化代码，从而将O(n^2)的复杂度降低到O(n).

常见问题分类：
		(1) 对于一个序列，用两个指针维护一段区间
		(2) 对于两个序列，维护某种次序，比如归并排序中合并两个有序序列的操作

前提：

绝大部分情况下，要求数组有序



分类

- 左右端点指针
  - 二分查找
  - 暴力枚举中“从大到小枚举”（剪枝）
  - 有序数组
- 快慢指针
  - 判断链表是否有环
  - 读写指针，典型的是`删除重复元素`
- 固定间距指针
  - 一次遍历求链表的中点
  - 一次遍历求链表的倒数第K个元素
  - 固定窗口的滑动窗口

伪代码

```java
// 快慢指针
l = 0
r = 0
while 没有遍历完
  if 一定条件
    l += 1
  r += 1
return 合适的值
```

```java
// 左右端点指针
l = 0
r = n - 1
while l < r
  if 找到了
    return 找到的值
  if 一定条件1
    l += 1
  else if  一定条件2
    r -= 1
return 没找到
```

```java
// 固定间距指针
l = 0
r = k
while 没有遍历完
  自定义逻辑
  l += 1
  r += 1
return 合适的值
```



双指针代码模板

```java
for (int i = 0, j = 0; i < n; i ++ )
{
	while (j < i && check(i, j)) j ++ ;
	
	// 具体问题的逻辑
}
```



### 位运算

 求n的第k位数字: n >> k & 1
 返回n的最后一位1：lowbit(n) = n & -n

1. 取反（NOT）
2. 按位或（OR）
3. 按位异或（XOR）
4. 按位与（AND）
5. 移位: 是一个二元运算符，用来将一个二进制数中的每一位全部都向一个方向移动指定位，溢出的部分将被舍弃，而空缺的部分填入一定的值。



参考资料：

[Bit Twiddling Hacks (stanford.edu)](http://graphics.stanford.edu/~seander/bithacks.html#IntegerLogIEEE64Float)



#### 离散化

离散化的核心思想：将分布大却数量少(即稀疏)的数据进行集中化的处理，减少空间复杂度

其实个人理解就是hash的思想。



离散化的两种思想和代码实现

- 包含重复元素，并且相同元素离散化后也要相同

```java
public class Discretization {
    public static int lower_bound(int[] arr,int target){ //找到第一个大于等于x的数的位置
        int l=0;
        int r=arr.length;
        while (l<r){
            int mid=l+(r-l)/2;
            if(arr[mid]>=target){
                r=mid;
            }else{
                l=mid+1;
            }
        }
        return l==arr.length?-1:l;
    }
    public static int[] solve(int[] array){
        SortedSet<Integer> set=new TreeSet<Integer>();
        //利用TreeSet可以同时实现排序和去重 可以代替c++中的unique实现
        for(int i=0;i<array.length;++i){
            set.add(array[i]);
        }
        //Integer[] b=(Integer[])set.toArray();
        int[] b=new int[set.size()]; //将去重排序的数组赋值给b数组
        int ct=0;
        for(int cur:set){
            b[ct++]=cur;
        }
        for(int i=0;i<array.length;++i){
            array[i]=lower_bound(b,array[i])+1; //利用lower_bound找到该数值的排位(rank)
            //排名代表大小 越前越小 越后越大
        }
        //10000000,2,2,123123213离散化成2,1,1,3
        return array;
    }
    public static void main(String[] args) {
        int[] a={10000000,2,2,123123213};
        solve(a);
    }
}
```

- 包含重复元素，并且相同元素离散化后不相同
- 不包含重复元素，并且不同元素离散化后不同



```java
public class Discretization {
    static class Node implements Comparable<Node>{
        int rt;
        int idx;
        public Node(int rt, int idx) {
            this.rt = rt;
            this.idx = idx;
        }
        @Override
        public int compareTo(Node node) {
            return rt-node.rt;
        }
    }
    public static void work(int[] array){
        int[] rank=new int[array.length];
        Node[] nodes=new Node[array.length];
        for(int i=0;i<array.length;++i){
            nodes[i]=new Node(array[i],i); //传入数值和坐标
        }
        java.util.Arrays.sort(nodes); //排序 记得实现Comparable接口 以rt大小排序
        for(int i=0;i<array.length;++i){
            rank[nodes[i].idx]=i;
        }
        for(int i=0;i<array.length;++i){
            array[i]=rank[i]+1; 
        }
        //10000000,2,2,123123213 离散化成 3 1 2 4
    }
    public static void main(String[] args) {
        int[] a={10000000,2,2,123123213};
        work(a);
    }

}
```

y总C++模板

```c++
	vector<int> alls; // 存储所有待离散化的值
	sort(alls.begin(), alls.end()); // 将所有值排序
	alls.erase(unique(alls.begin(), alls.end()), alls.end());	// 去掉重复元素
	
	// 二分求出x对应的离散化的值
	int find(int x)
	{
		int l = 0, r = alls.size() - 1;
		while (l < r)
		{
			int mid = l + r >> 1;
			if (alls[mid] >= x) r = mid;
			else l = mid + 1;
		}
		return r + 1;
	}
```



#### 区间合并

用途：将有交集的两个区间合并



#### 方法

（1）按照区间左端点进行排序
（2）指针st和ed维护当前区间，对区间进行扫描，扫描的下一区间（有三种情况）与当前区间有如下关系

![](D:\DownloadAndData\private\Java\myNote\markdown\basic\1007_7115e53e3d-区间合并.png)



```c++
	// 将所有存在交集的区间合并
	void merge(vector<PII> &segs)
	{
		vector<PII> res;

		sort(segs.begin(), segs.end());

		int st = -2e9, ed = -2e9;
		for (auto seg : segs)
			if (ed < seg.first)
			{
				if (st != -2e9) res.push_back({st, ed});
				st = seg.first, ed = seg.second;
			}
			else ed = max(ed, seg.second);

		if (st != -2e9) res.push_back({st, ed});

		segs = res;
	}
```

```python
# intervals 形如 [[1,3],[2,6]...]
def merge(intervals):
    if not intervals: return []
    # 按区间的 start 升序排列
    intervals.sort(key=lambda intv: intv[0])
    res = []
    res.append(intervals[0])
    for i in range(1, len(intervals)):
        curr = intervals[i]
        # res 中最后一个元素的引用
        last = res[-1]
        if curr[0] <= last[1]:
            # 找到最大的 end
            last[1] = max(last[1], curr[1])
        else:
            # 处理下一个待合并区间
            res.append(curr)
    return res
```





## 基础数据结构3节

### 第一节核心内容（已刷一遍）

单调栈、单调队列以及KMP算法，如何从朴素的算法优化出来的

链表与**邻接表**（邻接表的具体含义是啥？）

栈与队列：单调队列、单调栈（**队列与栈的基本操作有哪些，如何用数组实现队列与栈**）

kmp算法和核心的next数组

代码实现的问题

https://www.bilibili.com/video/BV1jb411V78H?from=search&seid=789489790236522154



### 第二节内容

### Tries树

0:00:00 ----26:00:00

概念：Trie树，又称字典树，单词查找树或者前缀树，是一种用于快速检索的多叉树结构，如英文字母的字典树是一个26叉树，数字的字典树是一个10叉树。

用处：高效的存储和查找字符串集合的数据结构



模板

```c++
1. Trie树

	int son[N][26], cnt[N], idx;
	// 0号点既是根节点，又是空节点
	// son[][]存储树中每个节点的子节点
	// cnt[]存储以每个节点结尾的单词数量

	// 插入一个字符串
	void insert(char *str)
	{
		int p = 0;
		for (int i = 0; str[i]; i ++ )
		{
			int u = str[i] - 'a';
			if (!son[p][u]) son[p][u] = ++ idx;
			p = son[p][u];
		}
		cnt[p] ++ ;
	}

	// 查询字符串出现的次数
	int query(char *str)
	{
		int p = 0;
		for (int i = 0; str[i]; i ++ )
		{
			int u = str[i] - 'a';
			if (!son[p][u]) return 0;
			p = son[p][u];
		}
		return cnt[p];
	}
```



### 应用场景

- 字符串检索

- 字符串最长公共前缀

- 字符串搜索的前缀匹配

- 作为其他数据结构和算法的辅助结构
- 词频统计
- 排序



###  并查集

0:26:00 

定义：并查集(Disjoint-Set)是一种可以动态维护若干个不重叠的集合，并支持*合并*与*查询*两种操作的一种数据结构。



基本操作：

1. 合并(Union/Merge)：将两个集合合并。
2. 查询(Find/Get)：询问两个元素是否在集合当中。

在近乎O(1)的复杂度完成以上两个操作



应用场景：

- 网络连接判断
- 变量名等同性（类似于指针的概念）



并查集代码的核心是：find

朴素并查集

```c++
int p[N]; //存储每个点的祖宗节点

		// 返回x的祖宗节点
		int find(int x)
		{
			if (p[x] != x) p[x] = find(p[x]);
			return p[x];
		}

		// 初始化，假定节点编号是1~n
		for (int i = 1; i <= n; i ++ ) p[i] = i;

		// 合并a和b所在的两个集合：
		p[find(a)] = find(b);
```

维护size的并查集：

```c++
int p[N], size[N];
		//p[]存储每个点的祖宗节点, size[]只有祖宗节点的有意义，表示祖宗节点所在集合中的点的数量

		// 返回x的祖宗节点
		int find(int x)
		{
			if (p[x] != x) 
                // 路径压缩
                p[x] = find(p[x]);
			return p[x];
		}

		// 初始化，假定节点编号是1~n
		for (int i = 1; i <= n; i ++ )
		{
			p[i] = i;
			size[i] = 1;
		}

		// 合并a和b所在的两个集合：
		p[find(a)] = find(b);
		size[b] += size[a];
```

维护到祖宗节点距离的并查集

```c++
int p[N], d[N];
		//p[]存储每个点的祖宗节点, d[x]存储x到p[x]的距离

		// 返回x的祖宗节点
		int find(int x)
		{
			if (p[x] != x)
			{
				int u = find(p[x]);
				d[x] += d[p[x]];
				p[x] = u;
			}
			return p[x];
		}

		// 初始化，假定节点编号是1~n
		for (int i = 1; i <= n; i ++ )
		{
			p[i] = i;
			d[I] = 0;
		}

		// 合并a和b所在的两个集合：
		p[find(a)] = find(b);
		d[find(a)] = distance; // 根据具体问题，初始化find(a)的偏移量
```



优化方法：

- 按秩（zhi,第四声）合并-----每次将比较矮的树，接到比较高的树下面
- 路径压缩（这种比较多，通过路径压缩后，几乎可以看成O(1)的时间复杂度）----遍历一次后，将路径上所有节点都接到根节点下面去

参考资料：https://www.cnblogs.com/MrSaver/p/9607552.html



### 堆

01:10:00 开始讲

堆是一个完全二叉树

完全二叉树：除了最下面一层，树都是满的，最后一层从左到右排列



存储：一维数组存储,x的左儿子：2x；x的右儿子：2x+ 1



小顶堆

大顶堆

通过down和up操作来实现以下五个操作 

- 如何手写一个堆

  - 插入一个数: `heap[++size] =x; up(size);`

  - 求集合当中的最小值:`heap[1]`;

  - 删除最小值:`heap[1]= heap[size];size--;down(1);`

  - 删除任意一个数:`heap[k] = heap[size];size--;down(k);up(k);`

  - 修改任意一个数:`heap[k] = x; down(k);up(k);`




```c++
	// h[N]存储堆中的值, h[1]是堆顶，x的左儿子是2x, 右儿子是2x + 1
	// ph[k]存储第k个插入的点在堆中的位置
	// hp[k]存储堆中下标是k的点是第几个插入的
	int h[N], ph[N], hp[N], size;

	// 交换两个点，及其映射关系
	void heap_swap(int a, int b)
	{
		swap(ph[hp[a]],ph[hp[b]]);
		swap(hp[a], hp[b]);
		swap(h[a], h[b]);
	}

	void down(int u)
	{
		int t = u;
		if (u * 2 <= size && h[u * 2] < h[t]) t = u * 2;
		if (u * 2 + 1 <= size && h[u * 2 + 1] < h[t]) t = u * 2 + 1;
		if (u != t)
		{
			heap_swap(u, t);
			down(t);
		}
	}

	void up(int u)
	{
		while (u / 2 && h[u] < h[u / 2])
		{
			heap_swap(u, u / 2);
			u >>= 1;
		}
	}
	
	// O(n)建堆
	for (int i = n / 2; i; i -- ) down(i);
```



### 第三节内容

### hash表（00:00:00---01:12:00）

离散化是一种特殊的hash方式，该如何理解？

存储结构：开放寻址法、拉链法



字符串哈希方式：字符串前缀哈希法 

场景：

可以去看java String的hashcode方法



STL(01:16:00---02:12:00)





## 搜索与图论3节（3*2 =6天）



## 数学知识4节（4*2 = 8天）



## 动态规划3节



## 贪心2节



## 时空复杂度1节



## 习题课8节



学算法到底学的是什么东西呢？除了学到具体的算法，我觉得更多的是体会算法的思想和思路，代码的优化过程。以及编码习惯等等。