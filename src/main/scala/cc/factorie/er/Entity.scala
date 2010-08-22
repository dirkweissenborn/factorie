/* Copyright (C) 2008-2010 Univ of Massachusetts Amherst, Computer Science Dept
   This file is part of "FACTORIE" (Factor graphs, Imperative, Extensible)
   http://factorie.cs.umass.edu, http://code.google.com/p/factorie/
   This software is provided under the terms of the Eclipse Public License 1.0
   as published by http://www.opensource.org.  For further information,
   see the file `LICENSE.txt' included with this distribution. */

package cc.factorie.er
import scala.collection.mutable.{ArrayStack,HashSet,HashMap,ListBuffer}
import cc.factorie._

// Define attributes and entities

/** A generic Attribute */
trait AttributeOf[E] extends Variable {
  /** The entity that is described by this attribute. */
  def attributeOwner: E
  /** Print the owner of the attribute before the rest of its toString representation. */
  override def toString = attributeOwner.toString+":"+super.toString
}
// I considered changing this trait name because of concerns that it is too generic.  For example it might be desired for coref.
// But now I think it is good for the "entity-relationship" package.  Users can always specify it with "er.Entity", which isn't bad.
// TODO Note that it is hard to subclass one of these, which seems sad.  
//  For example, users might want to subclass the pre-packaged entities in cc.factorie.application.  Think about this some more.
/** A trait for entities that have attributes.  Provides an inner trait 'Attribute' for its attribute classes. */
trait Entity[This<:Entity[This] with Variable] extends Variable {
  this: This =>
    //type VariableType = This
    type EntityType = This
  def thisEntity: This = this
  /** Sub-trait of cc.factorie.er.AttributeOf that has a concrete implementation of 'attributeOwner'. */
  trait Attribute extends cc.factorie.er.AttributeOf[This] {
    //type VariableType <: Attribute
    class GetterClass extends AttributeGetter[Attribute,This]
    def attributeOwner: This = thisEntity
  }
  /** Consider removing this.  Not sure if it should go here or outside the Entity. */
  class SymmetricFunction(initval:This, val get:This=>SymmetricFunction) extends RefVariable[This](initval) {
    type EntityType = This //with GetterType[This]
    def this(g:This=>SymmetricFunction) = this(null.asInstanceOf[This], g)
    override def set(newValue:This)(implicit d:DiffList) = {
      if (value != null) get(value)._set(null.asInstanceOf[This]) // Why is this cast necessary?
      super.set(newValue)(d)
      if (newValue != null) get(newValue)._set(thisEntity)
    }
    protected def _set(newValue:This)(implicit d:DiffList) = super.set(newValue)(d)
  }
}

