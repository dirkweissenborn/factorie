/* Copyright (C) 2008-2014 University of Massachusetts Amherst.
   This file is part of "FACTORIE" (Factor graphs, Imperative, Extensible)
   http://factorie.cs.umass.edu, http://github.com/factorie
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */


package cc.factorie.directed

import cc.factorie.directed.factor.{DiscreteGeneratingFactor, DirectedFamily2}
import cc.factorie.variable.{DiscreteValue, DiscreteSeqVar, DiscreteVar}
import scala.util.Random
import scala.collection.mutable


object MultinomialFromSeq extends DirectedFamily2[DiscreteVar, DiscreteSeqVar] {

  case class Factor(override val _1: DiscreteVar, override val _2: DiscreteSeqVar) extends super.Factor(_1, _2) with DiscreteGeneratingFactor {
    def pr(child: DiscreteVar#Value, seq: DiscreteSeqVar#Value) = seq.count(_.intValue == child.intValue).toDouble / seq.size
    override def prValue(intValue: Int): Double = _2.count(_.intValue == intValue).toDouble / _2.size
    def sampledValue(p1: DiscreteSeqVar#Value)(implicit random: Random) = _1.domain.apply(p1(random.nextInt(p1.size)).intValue).asInstanceOf[DiscreteVar#Value]
  }

  def newFactor(a: DiscreteVar, b: DiscreteSeqVar) = {
    if (a.domain.size != b.domain.elementDomain.size) throw new Error("Discrete child domain size different from parent element domain size.")
    Factor(a, b)
  }
}

object PlatedMultinomialFromSeq extends DirectedFamily2[DiscreteSeqVar, DiscreteSeqVar] {

  case class Factor(override val _1: DiscreteSeqVar, override val _2: DiscreteSeqVar) extends super.Factor(_1, _2) {
    def prValue(intValue: Int, parent: DiscreteSeqVar): Double = parent.count(_.intValue == intValue).toDouble / parent.size

    def pr(children: DiscreteSeqVar#Value, parent: DiscreteSeqVar#Value) = {
      val counts = mutable.HashMap[Int, Double]()
      parent.foreach(value => counts += value.intValue -> (1.0 + counts.getOrElse(value.intValue, 0.0)))
      val size = parent.size.toDouble
      children.foldLeft(1.0)((acc, child) => acc * counts(child.intValue) / size)
    }
    def sampledValue(p1: DiscreteSeqVar#Value)(implicit random: Random) = {
      val clone = IndexedSeq[DiscreteValue]()
      (0 until _1.length).foreach(_ => {
        val dom = _1.domain.elementDomain
        clone.+:(dom(p1(random.nextInt(p1.size)).intValue))
      })
      clone.asInstanceOf[DiscreteSeqVar#Value]
    }
  }

  def newFactor(a: DiscreteSeqVar, b: DiscreteSeqVar) = {
    if (a.domain.elementDomain.size != b.domain.elementDomain.size) throw new Error("Discrete child domain size different from parent element domain size.")
    Factor(a, b)
  }
}