# 推荐系统系列（一）：不到百行代码实现基于Spark的ItemCF计算
## 引言
信息大爆炸的互联网年代，推荐系统是帮助人们更高效获取信息的手段之一。从淘宝天猫的商品推荐到头条的信息流推荐，再到短视频推荐，推荐系统已经渗透到我们生活的方方面面。作为公众号的开篇系列，我们将分享关于推荐系统的技术文章从传统的协同过滤，到深度学习技术在推荐领域的应用，帮助读者更全面深入地了解推荐系统。

协同过滤是推荐系统最基础的算法，它可以简单分为User-based CF和Item-based CF。ItemCF的核心思想是选择当前用户偏好的物品的相似物品作为推荐结果。而UserCF是选择当前用户的相似用户偏好的物品作为这个用户的推荐结果。这篇文章将介绍如何基于Spark用不到一百行的代码实现相似物品的计算。

## 数据准备
推荐系统是由数据驱动的，在实际企业工作中，用户行为数据存储在数据仓库中。假设数据仓库上有一张用户行为日志表：t_user_interaction，它的DDL如下：

```sql 
CREATE TABLE t_user_interaction(
  `user_id` string COMMENT 'User ID', 
  `item_id` string COMMENT 'Item Id',
  `action_time` bigint COMMENT '动作发生的时间')
PARTITIONED BY ( 
  dt bigint)
```

通过Spark的SQL引擎很容易获取我们需要的数据：
```scala
val sql =
  s"""
     |select
     |user_id,
     |item_id
     |from t_user_interaction
     |where dt>=${param.startDt} and dt<=${param.endDt}
   """.stripMargin
   
val interactions = spark.sql(sql)
      .rdd
      .map(r => {
        val userId = r.getAs[String]("user_id")
        val itemId = r.getAs[String]("item_id")
        (userId, itemId)
      })
```
这里我们设置了两个参数：`startDt`,`endDt`，即一个滑动时间窗的开始时间和结束时间。实际生产环境，用户的行为在连续不断产生的，线上会不间断的收集这些行为日志，然后按一定时间窗，比如一个小时，来保存数据。ItemCF的计算任务也需要按一定时间滑动窗口周期运行，因为会不断有新的物品出现，系统需要尽可能快地计算出新物品的相似物品，才能在用户对新物品产生新的行为后尽快作出响应。

## 相似度计算
物品的相似度计算有很多公式，这里为了简单更易入门，我们以最常用的余弦相似度为例。
$$sim_{X,Y}=\frac{XY}{||X||||Y||}=\frac{ \sum_{i=1}^n(x_iy_i)}{\sqrt{\sum_{i=1}^n(x_i)^2}*\sqrt{\sum_{i=1}^n(y_i)^2}}$$

公式中$x_i$表示第$i$个用户对物品$x$的评分，$y_i$同理。
在实际生产中用户的显式评分数据很少，大多是一些隐式反馈（implicit feedback）数据，比如点击或者购买，所以我们用0或者1来表示用户对物品的偏好程度。以新闻推荐为例，1可以是用户点击了一篇文章，0表示给曝光了某篇文章但是用户没点击，或者用户根本没见过这篇文章。上面的公式可以拆解成分子和分母两部分：分子可以理解成是同时点击了文章$x$和文章$y$的用户数。分母包含了点击了文章$x$的用户数和点击了文章$y$的用户数。

首先，我们计算好每个文章的点击数备用。
```scala
// 统计每个文章的点击用户数
val itemClickCnt = interactions.map {
  case (_, itemId) => (itemId, 1)
}.reduceByKey((a, b) => a + b)
```

接着计算每两篇文章同时被点击的次数。假设总共有$N$篇文章，两两的组合数有$N*(N-1)/2$。直接的思路是拿到每个物品的点击用户列表，然后两两组合，求出两个点击用户列表的交集。这个思路比较容易理解，但是面临计算量太大，任务可能无法完成的问题。比如$N=100000$,就需要至少数十亿量级的计算。在生产环境，文章的数量常常不止十万的量级，其他业务场景下，物品的数量可能有百万级甚至更多。实际上并非所有文章组合都有共现，即有些文章组合从来没有被同一个用户点击过，这些文章组合的相似度为0，对后续的推荐没有作用，可以忽略掉。因此，我们可以只计算至少被一个用户同时点击过的文章组合。共现的基础是一个用户点了多篇文章，类似用Map-Reduce思想实现Word-Counter的方法，先收集每个用户的点击文章列表，然后罗列出两两文章的组合，再统计这些组合出现的次数。

```scala
// 统计每两个Item被一个用户同时点击出现的次数
val coClickNumRdd = interactions.groupByKey
  .filter {
    case (_, items) =>
      items.size > 1 && items.size < param.maxClick // 去掉点击数特别多用户，可能是异常用户
  }
  .flatMap {
    case (_, items) =>
      (for {x <- items; y <- items} yield ((x, y), 1))
  }
  .reduceByKey((a, b) => a + b)
  .filter(_._2 >= param.minCoClick) // 限制最小的共现次数
```
注意把点击次数特别多的用户过滤掉，这些用户可能是网络的一些爬虫，会污染数据。同时，这个操作也解决了数据倾斜导致计算耗时太高或无法完成的问题。（数据倾斜是Spark计算任务常见的问题，可以理解为由于数据分布的不均匀，某些子任务计算耗时太高或者一直无法完成，导致整个任务耗时很长或者无法完成。）另外，还需要限制文章最小的共现次数，如果A和B两篇文章只是被一个用户同时点击了，不管计算出来的相似分数多高都不足以作为相似的充分理由，很有可能只是偶然发生的。一般来说，被更多用户同时点击，相似的分数会更加置信。

通过上面两步的操作，我们就完成了分子分母所需元素的计算。下面将他们合起来就可以计算相似度了。
```scala
val similarities = coClickNumRdd.map {
      case ((x, y), clicks) =>
        (x, (y, clicks))
    }.join(itemClickNumRdd)
      .map {
        case (x, ((y, clicks), xClickCnt)) =>
          (y, (clicks, x, xClickCnt))
      }.join(itemClickNumRdd)
      .map {
        case (y, ((clicks, x, xClickCnt), yClickCnt)) =>
          val cosine = clicks / math.sqrt(xClickCnt * yClickCnt)
          (x, y, cosine)
      }
```

得到物品之间的相似度后做一个简单的排序，截取最相似的$K$个物品，来作为线上的推荐的数据。
到这里相似物品的计算过程就完成了，完整的代码可以在GitHub上找到。[GitHub链接](https://github.com/Play-With-AI/recommender-system)

## 总结
这篇文章用不到一百行的代码实现了大数据场景下真实可用的ItemCF算法。读者可以稍作修改就能应用于实际的业务。限于篇幅，很多细节并没有详细展开，比如不同相似度公式的比较，数据倾斜问题，在后续的文章里，我们将做相应的补充。ItemCF是推荐系统最基本最简单但也不可或缺的算法，后续我们会继续分享其他推荐算法的原理和实现。

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>