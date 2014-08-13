package cc.factorie.nn.sentiment

import java.io.File

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.parsing.combinator.RegexParsers

/**
 * Created by diwe01 on 31.07.14.
 */
object LoadPTB extends RegexParsers {

  //Example: (3 (2 It) (4 (4 (2 's) (4 (3 (2 a) (4 (3 lovely) (2 film))) (3 (2 with) (4 (3 (3 lovely) (2 performances)) (2 (2 by) (2 (2 (2 Buy) (2 and)) (2 Accorsi))))))) (2 .)))
  def sentimentPTBFromFile(f:File) = {
    Source.fromFile(f).getLines().map(treeString => {
      //HACK
      id = 0
      sentimentPTPFromString(treeString)
    })
  }


  def sentimentPTPFromString(treeString: String): SentimentPTree = {
    parseAll(tree, treeString) match {
      case Success(result, _) => new SentimentPTree(result.sortBy(_.id).toArray[SentimentPNode])
      case _ => throw new IllegalArgumentException(s"$treeString is not in sentiment PTB format!")
    }
  }

  private var id = 0

  val score:Parser[Int] = "[0-4]".r ^^ { case s => s.toInt}
  val node:Parser[ArrayBuffer[SentimentPNode]] = "(" ~> score ~ "[^()]+".r  <~ ")" ^^{
    case s ~ label =>
      id+=1
      ArrayBuffer(new SentimentPNode(s,id-1,-1,-1,label))
  }
  val tree:Parser[ArrayBuffer[SentimentPNode]] = node | "(" ~> score ~ tree ~ tree <~ ")" ^^ {
    case s ~ c1 ~ c2 =>
      val r = c1 ++ c2
      r.prepend(new SentimentPNode(s,id,c1.head.id,c2.head.id))
      id+=1
      r
  }
}

case class SentimentPNode(score:Int, id:Int, c1:Int,c2:Int, var label:String="")

class SentimentPTree(val nodes:Array[SentimentPNode]) {
  //create lookup
  val _parents:Map[Int,Int] = nodes.foldLeft(Map[Int,Int]())((acc,n) => if(n.c1>=0) acc + (nodes(n.c1).id -> n.id) + (nodes(n.c2).id -> n.id) else acc).toMap
  def parent(n:SentimentPNode) = _parents.get(n.id).map(nodes)
  val root = nodes.last
}
