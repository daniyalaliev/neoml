/* Copyright © 2023-2024 ABBYY

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

#include "common.h"
#pragma hdrstop

#include <cmath>

#include "Graph.h"
#include "GELUOptimizer.h"
#include <NeoML/Dnn/Layers/Onnx/OnnxEltwiseLayer.h>

using namespace NeoML;

namespace NeoOnnx {

namespace optimization {

// Checks if layer is an ONNX eltwise layer with the given scalar as one of its 2 inputs
// If that's the case selects both eltwise and scalar data layers and returns data from the other input
// Otherwise returns empty data (Layer == nullptr, Index == NotFound)
static CLayerOutput<> selectEltwiseByScalar( COnnxEltwiseLayer::TOperation expectedOp, float expectedScalar,
	CGraph& graph, CBaseLayer& layer )
{
	NeoAssert( !graph.IsLayerSelected( layer ) );

	if( graph.GetInputCount( layer ) != 2 || graph.GetOutputCount( layer ) != 1 ) {
		return CLayerOutput<>();
	}

	COnnxEltwiseLayer* eltwise = dynamic_cast<COnnxEltwiseLayer*>( &layer );
	if( eltwise == nullptr || eltwise->GetOperation() != expectedOp ) {
		return CLayerOutput<>();
	}

	for( int i = 0; i < 2; ++i ) {
		CLayerOutput<CDataLayer> dataInput = graph.GetConnectedOutput<CDataLayer>( layer, i );
		if( dataInput.Layer == nullptr ) {
			continue;
		}

		// We don't consider the case when both input of this layer are CDataLayers because that's invalid for GELU
		CPtr<CDnnBlob> blob = dataInput.Layer->GetBlob();
		if( blob->GetDataSize() != 1 || blob->GetDataType() != CT_Float ||
			std::fabs( expectedScalar - blob->GetData().GetValue() ) > 1e-4f )
		{
			return CLayerOutput<>();
		}
		
		graph.SelectLayer( layer );
		graph.SelectLayer( *dataInput.Layer );

		return graph.GetConnectedOutput<>( layer, 1 - i );
	}

	return CLayerOutput<>();
}

// Selects layer that divides its input by half
static CLayerOutput<> selectHalfLayer( CGraph& graph, CBaseLayer& layer )
{
	CLayerOutput<> result = selectEltwiseByScalar( COnnxEltwiseLayer::TOperation::Mul, 0.5f, graph, layer );
	if( result.Layer == nullptr ) {
		result = selectEltwiseByScalar( COnnxEltwiseLayer::TOperation::Div, 2.f, graph, layer );
	}
	return result;
}

// Selects layer that adds 1 to its input
static CLayerOutput<> selectAddOneLayer( CGraph& graph, CBaseLayer& layer )
{
	CLayerOutput<> result = selectEltwiseByScalar( COnnxEltwiseLayer::TOperation::Add, 1.f, graph, layer );
	if( result.Layer == nullptr ) {
		result = selectEltwiseByScalar( COnnxEltwiseLayer::TOperation::Sub, -1.f, graph, layer );
	}
	return result;
}

// Selects layer that divides its input by sqrt(2)
static CLayerOutput<> selectDivSqrt2Layer( CGraph& graph, CBaseLayer& layer )
{
	const float sqrt2 = std::sqrt( 2.f );
	CLayerOutput<> result = selectEltwiseByScalar( COnnxEltwiseLayer::TOperation::Div, sqrt2, graph, layer );
	if( result.Layer == nullptr ) {
		result = selectEltwiseByScalar( COnnxEltwiseLayer::TOperation::Mul, 1.f / sqrt2, graph, layer );
	}
	return result;
}

// Replaces graph selection with CGELULayer
static void replaceSelectionWithGELU( CGraph& graph, CLayerOutput<>& geluInput, CBaseLayer& prevOutput )
{
	CPtr<CGELULayer> gelu = new CGELULayer( graph.MathEngine() );
	gelu->SetName( graph.GetUniqueName( "GELU" ) );
	gelu->SetCalculationMode( CGELULayer::CM_Precise );
	graph.AddLayer( *gelu );
	graph.Connect( *gelu, 0, *geluInput.Layer, geluInput.Index );
	graph.SwitchOutputs( prevOutput, 0, *gelu, 0 );
	graph.DeleteSelectedLayers();
}

// Detects and replaces GELU generated by newer versions of PyTorch
// -+- -> Div(Sqrt2) -> Erf -> Add(1) -> Mul -> Div(2) ->
//  |                                     |
//  +-------------------------------------+
// Returns false if replacement has failed
static bool replaceNewVerGELU( CGraph& graph, CBaseLayer& halfLayer )
{
	for( int addOneIndex = 0; addOneIndex < 2; ++addOneIndex ) {
		graph.ClearSelection();

		COnnxEltwiseLayer* eltwiseMul = dynamic_cast<COnnxEltwiseLayer*>( selectHalfLayer( graph, halfLayer ).Layer );
		if( eltwiseMul == nullptr || eltwiseMul->GetOperation() != COnnxEltwiseLayer::TOperation::Mul || 
			graph.GetInputCount( *eltwiseMul ) != 2 )
		{
			continue;
		}
		graph.SelectLayer( *eltwiseMul );

		CBaseLayer* addOneLayer = graph.GetConnectedOutput( *eltwiseMul, addOneIndex ).Layer;
		NeoAssert( addOneLayer != nullptr );

		CErfLayer* erfLayer = dynamic_cast<CErfLayer*>( selectAddOneLayer( graph, *addOneLayer ).Layer );
		if( erfLayer == nullptr || graph.GetOutputCount( *erfLayer ) != 1 || graph.GetInputCount( *erfLayer ) != 1 ) {
			continue;
		}
		graph.SelectLayer( *erfLayer );

		CBaseLayer* divSqrt2 = graph.GetConnectedOutput( *erfLayer, 0 ).Layer;
		NeoAssert( divSqrt2 != nullptr );
		CLayerOutput<> divSqrt2Input = selectDivSqrt2Layer( graph, *divSqrt2 );

		CLayerOutput<> geluData = graph.GetConnectedOutput<>( *eltwiseMul, 1 - addOneIndex );
		if( geluData.Layer != nullptr && divSqrt2Input == geluData ) {
			replaceSelectionWithGELU( graph, geluData, halfLayer );
			return true;
		}
	}

	return false;
}

// Detects and replaces GELU generated by older version of PyTorch
// -+- -> Div(Sqrt2) -> Erf -> Add(1) -> Mul ->
//  |                                     |
//  +------------------------> Div(2) ----+
static bool replaceOldVerGELU( CGraph& graph, CBaseLayer& lastLayer )
{
	for( int addOneIndex = 0; addOneIndex < 2; ++addOneIndex ) {
		graph.ClearSelection();

		COnnxEltwiseLayer* eltwiseMul = dynamic_cast<COnnxEltwiseLayer*>( &lastLayer );
		if( eltwiseMul == nullptr || eltwiseMul->GetOperation() != COnnxEltwiseLayer::TOperation::Mul || 
			graph.GetInputCount( *eltwiseMul ) != 2 )
		{
			continue;
		}
		graph.SelectLayer( *eltwiseMul );

		CBaseLayer* addOneLayer = graph.GetConnectedOutput( *eltwiseMul, addOneIndex ).Layer;
		NeoAssert( addOneLayer != nullptr );

		CErfLayer* erfLayer = dynamic_cast<CErfLayer*>( selectAddOneLayer( graph, *addOneLayer ).Layer );
		if( erfLayer == nullptr || graph.GetOutputCount( *erfLayer ) != 1 || graph.GetInputCount( *erfLayer ) != 1 ) {
			continue;
		}
		graph.SelectLayer( *erfLayer );

		CBaseLayer* divSqrt2 = graph.GetConnectedOutput( *erfLayer, 0 ).Layer;
		NeoAssert( divSqrt2 != nullptr );
		CLayerOutput<> divSqrt2Input = selectDivSqrt2Layer( graph, *divSqrt2 );

		CBaseLayer* halfLayer = graph.GetConnectedOutput( *eltwiseMul, 1 - addOneIndex ).Layer;
		NeoAssert( halfLayer != nullptr );

		CLayerOutput<> geluData = selectHalfLayer( graph, *halfLayer );
		if( geluData.Layer != nullptr && divSqrt2Input == geluData ) {
			replaceSelectionWithGELU( graph, geluData, lastLayer );
			return true;
		}
	}

	return false;
}

int OptimizeGELU( CGraph& graph )
{
	int result = 0;

	CArray<CBaseLayer*> layers;
	graph.GetLayers( layers );
	for( CBaseLayer* layer : layers ) {
		if( !graph.HasLayer( layer ) ) {
			continue;
		}

		if( replaceNewVerGELU( graph, *layer ) || replaceOldVerGELU( graph, *layer ) ) {
			++result;
		}
	}

	graph.ClearSelection();
	return result;
}

} // namespace optimization

} // namespace NeoOnnx
