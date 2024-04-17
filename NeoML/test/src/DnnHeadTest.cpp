/* Copyright © 2024 ABBYY

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

#include <NeoML/Dnn/DnnHead.h>
#include <NeoML/Dnn/Layers/DnnHeadAdapterLayer.h>
#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

TEST( CDnnHeadTest, CDnnHeadInferenceAndLearnTest )
{
    //                      
    //                       +----------------+
    //                       |                |
    //                       |                v
    //[Source]  --->  [FullyConnected]     |------|
    //       \                             | HEAD | -> [Concat] ->[Loss]
    //        \ --->  [FullyConnected]     |------|
    //                       |                ^
    //                       |                |
    //                       +----------------+

    IMathEngine& mathEngine = MathEngine();
    CRandom random( 0 );
    CDnn dnn( random, mathEngine );

    CPtr<CHeadStruct> head = CDnnHead<
        CFullyConnectedLayer,
        CGELULayer,
        CFullyConnectedLayer,
        CReLULayer,
        CFullyConnectedLayer
    >(
        random, mathEngine,
        FullyConnected( 100 ),
        Gelu(),
        FullyConnected( 50 ),
        Relu(),
        FullyConnected( 1 )
    );

    CPtr<CSourceLayer> source = Source(dnn, "srcX");
    CPtr<CDnnBlob> dataBlob1 = CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 4, 2, 3, 10 });
    dataBlob1->Fill(1.f);
    source->SetBlob(dataBlob1);

    CPtr<CFullyConnectedLayer> fc1 = new CFullyConnectedLayer(MathEngine(), "fc1");
    dnn.AddLayer(*fc1);
    fc1->Connect(*source);
    fc1->SetNumberOfElements(5);

    CPtr<CDnnHeadAdapterLayer> head1 = new CDnnHeadAdapterLayer(MathEngine());
    head1->SetName("head1");
    head1->Connect(*fc1);
    head1->SetDnnHead(head);
    dnn.AddLayer(*head1);

    CPtr<CFullyConnectedLayer> fc2 = new CFullyConnectedLayer(MathEngine(), "fc2");
    dnn.AddLayer(*fc2);
    fc2->Connect(*source);
    fc2->SetNumberOfElements(5);

    CPtr<CDnnHeadAdapterLayer> head2 = new CDnnHeadAdapterLayer(MathEngine());
    head2->SetName("head2");
    head2->Connect(*fc2);
    head2->SetDnnHead(head);
    dnn.AddLayer(*head2);

    CPtr<CConcatChannelsLayer> concat = new CConcatChannelsLayer(MathEngine());
    dnn.AddLayer(*concat);
    concat->Connect(0 ,*head1, 0);
    concat->Connect(1, *head2, 0);

    CPtr<CSourceLayer> labels = Source(dnn, "srcY");
    CPtr<CDnnBlob> datablob2 = CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 1, 1, 1, 2 });
    datablob2->Fill(10.f);
    labels->SetBlob(datablob2);

    CPtr<CEuclideanLossLayer> loss = new CEuclideanLossLayer(MathEngine());
    loss->SetName("loss");
    dnn.AddLayer(*loss);
    loss->Connect(0, *concat, 0);
    loss->Connect(1, *labels, 0);

    CPtr<CDnnSimpleGradientSolver> solver = new CDnnSimpleGradientSolver(MathEngine());
    const float learningRate = 1e-4f;
    solver->SetLearningRate(learningRate);
    dnn.SetSolver(solver.Ptr());

    std::unique_ptr<IPerformanceCounters> counters(MathEngine().CreatePerformanceCounters());
    counters->Synchronise();
    for (int i = 0; i < 1000; ++i) {
        dnn.RunAndLearnOnce();
    }
    counters->Synchronise();
    std::cerr << "Loss: " << loss->GetLastLoss()
        << '\t' << "Train Time: " << (double((*counters)[0].Value) / 1000000) << " ms."
        << '\t' << "Peak.Mem: " << (double(MathEngine().GetPeakMemoryUsage()) / 1024 / 1024) << " MB"
        << '\n';

    EXPECT_NEAR(loss->GetLastLoss(), 0.f, 1e-3f);

    CString name = "new_file";
    {
        CArchiveFile archiveFile(name, CArchive::store, GetPlatformEnv());
        CArchive archive(&archiveFile, CArchive::SD_Storing);
        dnn.Serialize(archive);
    }

    {
        CArchiveFile archiveFile(name, CArchive::load, GetPlatformEnv());
        CArchive archive(&archiveFile, CArchive::SD_Loading);
        dnn.Serialize(archive);
    }

    dynamic_cast<CSourceLayer*>(dnn.GetLayer("srcX").Ptr())->SetBlob(dataBlob1);
    dynamic_cast<CSourceLayer*>(dnn.GetLayer("srcY").Ptr())->SetBlob(datablob2);
    OptimizeDnn(dnn);
    dnn.RunOnce();
}

TEST(CDnnHeadTest, DummyImplement)
{
    //                      
    //                       +----------------+
    //                       |                |
    //                       |                v
    //[Source]  --->  [FullyConnected]     |------|
    //       \                             | HEAD | -> [Concat] ->[Loss]
    //        \ --->  [FullyConnected]     |------|
    //                       |                ^
    //                       |                |
    //                       +----------------+

    IMathEngine& mathEngine = MathEngine();
    CRandom random(0);
    CDnn dnn(random, mathEngine);


    CPtr<CSourceLayer> source = Source(dnn, "srcX");
    CPtr<CDnnBlob> dataBlob1 = CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 4, 2, 3, 10 });
    dataBlob1->Fill(1.f);
    source->SetBlob(dataBlob1);

    CPtr<CFullyConnectedLayer> fc1 = new CFullyConnectedLayer(MathEngine(), "fc1");
    dnn.AddLayer(*fc1);
    fc1->Connect(*source);
    fc1->SetNumberOfElements(5);


    CPtr<CFullyConnectedLayer> fc2 = new CFullyConnectedLayer(MathEngine(), "fc2");
    dnn.AddLayer(*fc2);
    fc2->Connect(*source);
    fc2->SetNumberOfElements(5);

    CPtr<CFullyConnectedLayer> fullLayer1 = new CFullyConnectedLayer(MathEngine());
    fullLayer1->SetName("cfc1");
    dnn.AddLayer(*fullLayer1);
    fullLayer1->SetNumberOfElements(100);
    fullLayer1->Connect(0, *fc1);
    fullLayer1->Connect(1, *fc2);

    CPtr<CGELULayer> gelu1 = new CGELULayer(MathEngine());
    gelu1->SetName("gelu1");
    dnn.AddLayer(*gelu1);
    gelu1->Connect(0, *fullLayer1, 0);

    CPtr<CGELULayer> gelu2 = new CGELULayer(MathEngine());
    gelu2->SetName("gelu2");
    dnn.AddLayer(*gelu2);
    gelu2->Connect(0, *fullLayer1, 1);


    CPtr<CFullyConnectedLayer> fullLayer2 = new CFullyConnectedLayer(MathEngine());
    fullLayer2->SetName("cfc2");
    dnn.AddLayer(*fullLayer2);
    fullLayer2->SetNumberOfElements(50);
    fullLayer2->Connect(0, *gelu1);
    fullLayer2->Connect(1, *gelu2);

    CPtr<CGELULayer> relu1 = new CGELULayer(MathEngine());
    relu1->SetName("relu1");
    dnn.AddLayer(*relu1);
    relu1->Connect(0, *fullLayer2, 0);

    CPtr<CGELULayer> relu2 = new CGELULayer(MathEngine());
    relu2->SetName("relu2");
    dnn.AddLayer(*relu2);
    relu2->Connect(0, *fullLayer2, 1);

    CPtr<CFullyConnectedLayer> fullLayer3 = new CFullyConnectedLayer(MathEngine());
    fullLayer3->SetName("cfc3");
    fullLayer3->SetNumberOfElements(1);
    dnn.AddLayer(*fullLayer3);
    fullLayer3->Connect(0, *relu1);
    fullLayer3->Connect(1, *relu2);

    CPtr<CConcatChannelsLayer> concat = new CConcatChannelsLayer(MathEngine());
    dnn.AddLayer(*concat);
    concat->Connect(0, *fullLayer3, 0);
    concat->Connect(1, *fullLayer3, 1);

    CPtr<CSourceLayer> labels = Source(dnn, "srcY");
    CPtr<CDnnBlob> datablob2 = CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 1, 1, 1, 2 });
    datablob2->Fill(10.f);
    labels->SetBlob(datablob2);

    CPtr<CEuclideanLossLayer> loss = new CEuclideanLossLayer(MathEngine());
    loss->SetName("loss");
    dnn.AddLayer(*loss);
    loss->Connect(0, *concat, 0);
    loss->Connect(1, *labels, 0);

    CPtr<CDnnSimpleGradientSolver> solver = new CDnnSimpleGradientSolver(MathEngine());
    const float learningRate = 1e-4f;
    solver->SetLearningRate(learningRate);
    dnn.SetSolver(solver.Ptr());

    std::unique_ptr<IPerformanceCounters> counters(MathEngine().CreatePerformanceCounters());
    counters->Synchronise();
    for (int i = 0; i < 1000; ++i) {
        dnn.RunAndLearnOnce();
    }
    counters->Synchronise();
    std::cerr << "Loss: " << loss->GetLastLoss()
        << '\t' << "Train Time: " << (double((*counters)[0].Value) / 1000000) << " ms."
        << '\t' << "Peak.Mem: " << (double(MathEngine().GetPeakMemoryUsage()) / 1024 / 1024) << " MB"
        << '\n';

    EXPECT_NEAR(loss->GetLastLoss(), 0.f, 1e-3f);

    CString name = "new_file";
    {
        CArchiveFile archiveFile(name, CArchive::store, GetPlatformEnv());
        CArchive archive(&archiveFile, CArchive::SD_Storing);
        dnn.Serialize(archive);
    }

    {
        CArchiveFile archiveFile(name, CArchive::load, GetPlatformEnv());
        CArchive archive(&archiveFile, CArchive::SD_Loading);
        dnn.Serialize(archive);
    }

    dynamic_cast<CSourceLayer*>(dnn.GetLayer("srcX").Ptr())->SetBlob(dataBlob1);
    dynamic_cast<CSourceLayer*>(dnn.GetLayer("srcY").Ptr())->SetBlob(datablob2);
    OptimizeDnn(dnn);
    dnn.RunOnce();
}
