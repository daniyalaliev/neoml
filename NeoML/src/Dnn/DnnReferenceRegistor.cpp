/* Copyright @ 2024 ABBYY

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

#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

CDnnReferenceRegistor::CDnnReferenceRegistor() = default;

CDnnReferenceRegistor::CDnnReferenceRegistor(CDnn* _originalDnn) :
	learningState(false),
	referenceCounter(-1),
	originalDnn(_originalDnn)
{
	NeoAssert(_originalDnn != nullptr);
	if(originalDnn->referenceDnnRegistoror.referenceCounter++ == 0) {
		originalDnn->referenceDnnRegistoror.learningState = originalDnn->IsLearningEnabled();
	}
}

CDnnReferenceRegistor::~CDnnReferenceRegistor()
{
	if(referenceCounter == -1 && --(originalDnn->referenceDnnRegistoror.referenceCounter) == 0
		&& originalDnn->referenceDnnRegistoror.learningState)
	{
		originalDnn->EnableLearning();
	}
}

CDnnReferenceRegistor& CDnnReferenceRegistor::operator=(CDnnReferenceRegistor&& other) {
	if(this != &other) {
		learningState = other.learningState;
		referenceCounter = other.referenceCounter;
		originalDnn = other.originalDnn;

		other.originalDnn = nullptr;
		other.referenceCounter = 0;
		other.learningState = false;
	}
	return *this;
}

} // namespace NeoML
