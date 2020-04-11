package com.knowyou.rs.driver

import org.apache.spark.sql.SparkSession
import scopt.OptionParser


object ItemCFDriver {

  case class Parameters(
                         startDt: String = "",
                         endDt: String = "",
                         maxClick: Int = 100,
                         topK: Int = 100
                       ) extends Serializable

  val defaultParam: Parameters = Parameters()
  val parser: OptionParser[Parameters] = new OptionParser[Parameters](this.getClass.getSimpleName) {
    head("Main")
    opt[String]("startDs")
      .text(s"startDs: ${defaultParam.startDt}")
      .action((x, c) => c.copy(startDt = x))

    opt[String]("endDs")
      .text(s"endDs: ${defaultParam.endDt}")
      .action((x, c) => c.copy(endDt = x))

    opt[Int]("maxClick")
      .text(s"maxClick: ${defaultParam.maxClick}")
      .action((x, c) => c.copy(maxClick = x))

    opt[Int]("topK")
      .text(s"topK: ${defaultParam.topK}")
      .action((x, c) => c.copy(topK = x))
  }

  def main(args: Array[String]): Unit = {

    parser.parse(args, Parameters()) match {
      case Some(param) =>
        println(param)
        run(param)
      case _ =>
        println("parameter error!")
    }
  }

  def run(param: Parameters): Unit = {
    // 初始化Spark
    val spark = SparkSession
      .builder()
      .appName(this.getClass.getSimpleName)
      .enableHiveSupport()
      .getOrCreate()

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

    // 统计每个文章的点击用户数
    val itemClickNumRdd = interactions.map {
      case (_, itemId) => (itemId, 1)
    }.reduceByKey((a, b) => a + b)

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

    // TODO 取TOP，存储相似结果
  }
}
