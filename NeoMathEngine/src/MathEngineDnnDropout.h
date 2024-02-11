/* Copyright © 2017-2024 ABBYY Production LLC

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

#pragma once

#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoMathEngine/CrtAllocatedObject.h>

namespace NeoML {

// Dropout descriptor containing whole mask
struct CMaskDropoutDesc : public CDropoutDesc {
	explicit CMaskDropoutDesc( IMathEngine& mathEngine, float rate, bool isSpatial, bool isBatchwise,
		const CBlobDesc& input, const CBlobDesc& output, int seed );

	CBlobDesc Input; // input blob descriptor
	CBlobDesc Output; // output blob descriptor
	const float ForwardRate; // the probability that an element is not dropped out
	const bool IsSpatial; // indicates if whole channels are dropped out
	const bool IsBatchwise; // indicates if an element is dropped out of all objects in one batch at the same time
	// A blob that stores the dropout information for each element on the last run
	// Only used when learning
	CFloatHandleVar Mask;
};

// Dropout descriptor containing fixed memory for generating mask parts iteratively
struct CSeedDropoutDesc : public CDropoutDesc {
	explicit CSeedDropoutDesc( IMathEngine& mathEngine, float rate, bool isSpatial, bool isBatchwise,
		const CBlobDesc& input, const CBlobDesc& output, int seed );

	CBlobDesc Input; // input blob descriptor
	CBlobDesc Output; // output blob descriptor
	const float ForwardRate; // the probability that an element is not dropped out
	const bool IsSpatial; // indicates if whole channels are dropped out
	const bool IsBatchwise; // indicates if an element is dropped out of all objects in one batch at the same time
	// // seed that will be used later to generate mask
	const int seed;
	const unsigned threshold; // = (unsigned int)((double)desc.ForwardRate * UINT_MAX);
	const float value; // = 1.f / desc.ForwardRate;
	CFloatHandleVar Mask; // mask of fixed size, filled iteratively in dropout
	static constexpr int maskAlign = 4;
	static constexpr int cacheSize = 64;
};

} // namespace NeoML
