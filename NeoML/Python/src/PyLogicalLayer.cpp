/* Copyright © 2017-2021 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------*/

#include <common.h>
#pragma hdrstop

#include "PyLogicalLayer.h"

class CPyNotLayer : public CPyLayer {
public:
	explicit CPyNotLayer( CNotLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Not" );
		return pyConstructor( py::cast(this) );
	}
};

class CPyLessLayer : public CPyLayer {
public:
	explicit CPyLessLayer( CLessLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Less" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeLogicalLayer( py::module& m )
{
	py::class_<CPyNotLayer, CPyLayer>(m, "Not")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyNotLayer( *layer.Layer<CNotLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CNotLayer> notLayer = new CNotLayer( mathEngine );
			notLayer->SetName( FindFreeLayerName( dnn, "Not", name ).c_str() );
			dnn.AddLayer( *notLayer );
			notLayer->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyNotLayer( *notLayer, layer.MathEngineOwner() );
		}) )
	;
	
	py::class_<CPyLessLayer, CPyLayer>(m, "Less")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyLessLayer( *layer.Layer<CLessLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& firstLayer, int firstOutputNumber,
				const CPyLayer& secondLayer, int secondOutputNumber ) {
			py::gil_scoped_release release;
			CDnn& dnn = firstLayer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CLessLayer> less = new CLessLayer( mathEngine );
			less->SetName( FindFreeLayerName( dnn, "Less", name ).c_str() );
			dnn.AddLayer( *less );
			less->Connect( 0, firstLayer.BaseLayer(), firstOutputNumber );
			less->Connect( 1, secondLayer.BaseLayer(), secondOutputNumber );
			return new CPyLessLayer( *less, firstLayer.MathEngineOwner() );
		}) )
	;
}