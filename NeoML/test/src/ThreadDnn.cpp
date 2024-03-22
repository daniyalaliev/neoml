/* Copyright @ 2024 ABBYY Production LLC

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
#include <mutex>
#pragma hdrstop

#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

struct TaskParams {
    CDnn* net;
};

static void RunDnn(int threadIndex, void* params) {
    TaskParams* taskParams = static_cast<TaskParams*>(params);
    taskParams->net->RunOnce();
    //printf("I HATE THIS FUCKING SHIT");
}

static CDnn* CreateDnn(CRandom& random)
{
    CDnn* net = new CDnn(random, MathEngine());

    CPtr<CSourceLayer> dataLayer = Source(*net, "in");

    CPtr<CFullyConnectedLayer> fc1 = new CFullyConnectedLayer(MathEngine());
    fc1->SetNumberOfElements(50);
    fc1->SetName("fc1");
    fc1->Connect(0, *dataLayer);
    net->AddLayer(*fc1);

    CPtr<CDropoutLayer> dp1 = new CDropoutLayer(MathEngine());
    dp1->SetName("dp1");
    dp1->Connect(0, *fc1);
    net->AddLayer(*dp1);

    CPtr<CFullyConnectedLayer> fc2 = new CFullyConnectedLayer(MathEngine());
    fc2->SetName("fc2");
    fc2->SetNumberOfElements(20);
    fc2->Connect(*dp1);
    net->AddLayer(*fc2);

    CPtr<CDropoutLayer> dp2 = new CDropoutLayer(MathEngine());
    dp2->SetName("dp2");
    dp2->Connect(0, *fc2);
    net->AddLayer(*dp2);

    CPtr<CFullyConnectedLayer> fc3 = new CFullyConnectedLayer(MathEngine());
    fc3->SetName("fc3");
    fc3->SetNumberOfElements(10);
    fc3->Connect(0, *dp2);
    net->AddLayer(*fc3);

    CPtr<CSinkLayer> sink = new CSinkLayer(MathEngine());
    sink->SetName("sink");
    sink->Connect(0, *fc3);
    net->AddLayer(*sink);

    return net;
}

TEST(ThreadTest, DefaultOneThread)
{
    CRandom random(0x123);
    CDnn* net = CreateDnn(random);
    CPtr<CDnnBlob> dataBlob = CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 32, 200, 30, 100 });
    static_cast<CSourceLayer*>(net->GetLayer("in").Ptr())->SetBlob(dataBlob);

    std::unique_ptr<IPerformanceCounters> counters(MathEngine().CreatePerformanceCounters());
    counters->Synchronise();

    net->RunOnce();

    counters->Synchronise();
    std::cerr 
        << '\n' << "Time: " << (double((*counters)[0].Value) / 1000000) << " ms."
        << '\t' << "Peak.Mem: " << (double(MathEngine().GetPeakMemoryUsage()) / 1024 / 1024) << " MB \n";
    delete net;
}

TEST(ThreadTest, FourThreads)
{
    CRandom random(0x123);
    CDnn* net = CreateDnn(random);
    CPtr<CDnnBlob> dataBlob = CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 8, 200, 30, 100 });
    static_cast<CSourceLayer*>(net->GetLayer("in").Ptr())->SetBlob(dataBlob);

    int numOfThreads = 4;
    CArray<CDnn*> dnns;
    CObjectArray<CDnnBlob> blobs;
    CArray<TaskParams> taskParams;

    std::unique_ptr<IPerformanceCounters> counters(MathEngine().CreatePerformanceCounters());
    counters->Synchronise();

    taskParams.Add({ net });
    for (int i = 0; i < numOfThreads-1; ++i) {
        dnns.Add(net->CreateReferenceDnn());
        blobs.Add(CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 8, 200, 30, 100 }));
        taskParams.Add({ dnns[i] });
        static_cast<CSourceLayer*>(dnns[i]->GetLayer("in").Ptr())->SetBlob(blobs[i]);
    }

    IThreadPool* pool = CreateThreadPool(numOfThreads);
    for (int i = 0; i < numOfThreads; ++i) {
        pool->AddTask(i, RunDnn, &(taskParams[i]));
    }

    pool->WaitAllTask();
    counters->Synchronise();
    std::cerr
        << '\n' << "Time: " << (double((*counters)[0].Value) / 1000000) << " ms."
        << '\t' << "Peak.Mem: " << (double(MathEngine().GetPeakMemoryUsage()) / 1024 / 1024) << " MB \n";

    EXPECT_TRUE(CompareBlobs(
        *(static_cast<CSinkLayer*>(dnns[0]->GetLayer("sink").Ptr())->GetBlob()),
        *(static_cast<CSinkLayer*>(net->GetLayer("sink").Ptr())->GetBlob()))
    );

    delete pool;
    delete net;
    for (int i = 0; i < numOfThreads-1; ++i) {
        delete dnns[i];
    }
}

TEST(ThreadTest, FourDnns)
{
    int numOfThreads = 4;

    CArray<CDnn*> dnns;
    CObjectArray<CDnnBlob> blobs;
    CArray<TaskParams> taskParams;
    CArray<CRandom> randoms;

    std::unique_ptr<IPerformanceCounters> counters(MathEngine().CreatePerformanceCounters());
    counters->Synchronise();

    for (int i = 0; i < numOfThreads; ++i) {
        randoms.Add(CRandom(0x123));
        dnns.Add(CreateDnn(randoms[i]));
        blobs.Add(CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 8, 200, 30, 100 }));
        taskParams.Add({ dnns[i] });
        static_cast<CSourceLayer*>(dnns[i]->GetLayer("in").Ptr())->SetBlob(blobs[i]);
    }

    IThreadPool* pool = CreateThreadPool(4);
    for (int i = 0; i < numOfThreads; ++i) {
        pool->AddTask(i, RunDnn, &(taskParams[i]));
    }

    pool->WaitAllTask();
    counters->Synchronise();
    std::cerr
        << '\n' << "Time: " << (double((*counters)[0].Value) / 1000000) << " ms."
        << '\t' << "Peak.Mem: " << (double(MathEngine().GetPeakMemoryUsage()) / 1024 / 1024) << " MB \n";

    for (int i = 0; i < numOfThreads; ++i) {
        delete dnns[i];
    }
}