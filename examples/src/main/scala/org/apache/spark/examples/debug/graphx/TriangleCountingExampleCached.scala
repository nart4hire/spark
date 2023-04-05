/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
package org.apache.spark.examples.debug.graphx

// $example on$
import org.apache.spark.graphx.{GraphLoader, PartitionStrategy}
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * A vertex is part of a triangle when it has two adjacent vertices with an edge between them.
 * GraphX implements a triangle counting algorithm in the [`TriangleCount` object][TriangleCount]
 * that determines the number of triangles passing through each vertex,
 * providing a measure of clustering.
 * We compute the triangle count of the social network dataset.
 *
 * Note that `TriangleCount` requires the edges to be in canonical orientation (`srcId < dstId`)
 * and the graph to be partitioned using [`Graph.partitionBy`][Graph.partitionBy].
 *
 * Run with
 * {{{
 * bin/run-example graphx.TriangleCountingExample
 * }}}
 */
object TriangleCountingExampleCached {
  def main(args: Array[String]): Unit = {
    // Creates a SparkSession.
    val spark = SparkSession
      .builder
      .appName(s"${this.getClass.getSimpleName}")
      .getOrCreate()
    val sc = spark.sparkContext

    // $example on$
    // Load the edges in canonical order and partition the graph for triangle count
    val graph = GraphLoader.edgeListFile(sc, args(0), true) // "data/graphx/followers.txt"
      .partitionBy(PartitionStrategy.RandomVertexCut).cache()
    // Find the triangle count for each vertex
    val triCounts = graph.triangleCount().vertices.cache()
    // Trigger execution of the job
    triCounts.count()
    // Print the result
    // println(triCounts.collect().mkString("\n"))
    // $example off$
    spark.stop()
  }
}
// scalastyle:on println
