package cc.factorie.tutorial

import cc.factorie.directed._
import cc.factorie.variable._
import scala.collection.mutable.ArrayBuffer
import java.io.File
import cc.factorie.app.strings.{alphaSegmenter, Stopwords}

/**
 * Created by diwe01 on 30.06.14.
 */
object HLDA {
  implicit val model = DirectedModel()

  object TopicDomain extends TreePathDomain
  class TopicPath extends TreePathVar {
    override def domain = TopicDomain
  }

  val maxLevel = 10
  class Level(init:Int = 0) extends IntegerVariable(init)

  object WordDomain extends CategoricalDomain[String]
  class Word(string:String) extends CategoricalVariable(string) {
    def domain = WordDomain
    lazy val level = model.parentFactor(this).asInstanceOf[PlatedNestedCRP.Factor].level
    lazy val path = model.parentFactor(this).asInstanceOf[PlatedNestedCRP.Factor].path
    def z = path.value.valueAtLevel(level.intValue)
  }

  class Document(val file:String, val theta:ProportionsVar, strings:Seq[String]) {
    val words = strings.map(s => new Word(s))
  }


/*
  def main(args: Array[String]): Unit = {
    implicit val random = new scala.util.Random(0)
    val directories = if (args.length > 0) args.toList else List("12", "11", "10", "09", "08").take(1).map("/Users/mccallum/research/data/text/nipstxt/nips"+_)
    val mixtureProportions = ArrayBuffer[ProportionsVariable](ProportionsVariable.growableDense(WordDomain) ~ Dirichlet(beta))
    val phis = Mixture(mixtureProportions)
    val documents = new ArrayBuffer[Document]
    for (directory <- directories) {
      for (file <- new File(directory).listFiles; if file.isFile) {
        val theta = ProportionsVariable.growableDense() ~ Dirichlet(alphas)
        val tokens = alphaSegmenter(file).map(_.toLowerCase).filter(!Stopwords.contains(_)).toSeq
        val zs = new (tokens.length) :~ PlatedDiscrete(theta)
        documents += new Document(file.toString, theta, tokens) ~ PlatedDiscreteMixture(phis, zs)
      }
    }

    val collapse = new ArrayBuffer[Var]
    collapse += phis
    collapse ++= documents.map(_.theta)
    val sampler = new CollapsedGibbsSampler(collapse, model)

    for (i <- 1 to 20) {
      for (doc <- documents) sampler.process(doc.zs)
    }

  }*/

}
