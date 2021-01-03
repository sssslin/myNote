# acwing算法基础课刷视频进度



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

## 基础数据结构3节

第一节核心内容（已刷一遍）

单调栈、单调队列以及KMP算法，如何从朴素的算法优化出来的

链表与**邻接表**（邻接表的具体含义是啥？）

栈与队列：单调队列、单调栈（**队列与栈的基本操作有哪些，如何用数组实现队列与栈**）

kmp算法和核心的next数组



## 搜索与图论3节



## 数学知识4节



## 动态规划3节



## 贪心2节



## 时空复杂度1节



## 习题课8节

