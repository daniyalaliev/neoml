/* Copyright © 2017-2023 ABBYY Production LLC

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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

class NEOML_API CParameterLayer : public CBaseLayer {
    NEOML_DNN_LAYER(CParameterLayer)
public:
    CParameterLayer(IMathEngine& mathEngine, const char* name = nullptr) :
        CBaseLayer(mathEngine, name == nullptr ? "ParameterLayer" : name, true) { paramBlobs.SetSize(1); };

    void Serialize(CArchive& archive) override;
    void SetBlob(CDnnBlob* _blob);
    const CPtr<CDnnBlob>& GetBlob() const { return paramBlobs[0]; }
protected:
    void AllocateOutputBlobs();
    void Reshape() override;
    void RunOnce() override;
    void BackwardOnce() override;
    void LearnOnce() override;
};

} // namespace NeoML
