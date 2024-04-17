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

//TEST(ThreadTest, Four1Threads)
//{
//    CRandom random(0x123);
//    CDnn* net = CreateDnn(random);
//    CPtr<CDnnBlob> dataBlob = CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 8, 200, 30, 100 });
//    static_cast<CSourceLayer*>(net->GetLayer("in").Ptr())->SetBlob(dataBlob);
//
//    int numOfThreads = 4;
//    CArray<CDnn*> dnns;
//    CObjectArray<CDnnBlob> blobs;
//    CArray<TaskParams> taskParams;
//
//    std::unique_ptr<IPerformanceCounters> counters(MathEngine().CreatePerformanceCounters());
//    counters->Synchronise();
//
//    taskParams.Add({ net });
//    for (int i = 0; i < numOfThreads-1; ++i) {
//        CRandom rand(0x123);
//        dnns.Add(net->CreateReferenceDnn(rand));
//        blobs.Add(CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 8, 200, 30, 100 }));
//        taskParams.Add({ dnns[i] });
//        static_cast<CSourceLayer*>(dnns[i]->GetLayer("in").Ptr())->SetBlob(blobs[i]);
//    }
//
//    IThreadPool* pool = CreateThreadPool(numOfThreads);
//    for (int i = 0; i < numOfThreads; ++i) {
//        pool->AddTask(i, RunDnn, &(taskParams[i]));
//    }
//
//    pool->WaitAllTask();
//    counters->Synchronise();
//    std::cerr
//        << '\n' << "Time: " << (double((*counters)[0].Value) / 1000000) << " ms."
//        << '\t' << "Peak.Mem: " << (double(MathEngine().GetPeakMemoryUsage()) / 1024 / 1024) << " MB \n";
//
//    EXPECT_TRUE(CompareBlobs(
//        *(static_cast<CSinkLayer*>(dnns[0]->GetLayer("sink").Ptr())->GetBlob()),
//        *(static_cast<CSinkLayer*>(net->GetLayer("sink").Ptr())->GetBlob()))
//    );
//
//    delete pool;
//    delete net;
//    for (int i = 0; i < numOfThreads-1; ++i) {
//        delete dnns[i];
//    }
//}

TEST(ThreadTest, Four1Dnns)
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

static const int paragraphs_cnt = 100;
static const int batch_size = 4;
static const int numOfThreads = 4;
static const int numOfMeasures = 10;

static void initializeInput(CRandom& random, CPtr<CDnnBlob> blob, int lower, int higher)
{
    CPtr<CDnnUniformInitializer> uniformInitializer = new CDnnUniformInitializer(CRandom(0x123), 0, 50000);
    CArray<int> tempData;
    tempData.SetSize(blob->GetDataSize());

    int* data = tempData.GetPtr();
    for (int i = 0; i < tempData.Size(); ++i) {
        int num = (int)(uniformInitializer->Random()).Uniform(lower, higher);
        *data++ = num;
    }

    blob->CopyFrom(tempData.GetPtr());
}

TEST(ThreadTest, BertTest)
{
    CRandom random(0x123);
    CDnn bert(random, MathEngine());
    CArchiveFile file(".\\RobertaTextSeqmentationInference.dnn", CArchive::load);
    CArchive archive(&file, CArchive::SD_Loading);

    bert.Serialize(archive);
    //bert.SetInitializer()

    CPtr<CDnnBlob> input_ids = CDnnBlob::CreateTensor(MathEngine(), CT_Int, { 1, batch_size, 512, 1, 1, 1, 1 });
    initializeInput(CRandom(0x123), input_ids, 1, 2);
    static_cast<CSourceLayer*>(bert.GetLayer("input_ids").Ptr())->SetBlob(input_ids);

    CPtr<CDnnBlob> clsPositions = CDnnBlob::CreateTensor(MathEngine(), CT_Int, { 1, 1, 1, 1, 1, 1, paragraphs_cnt });
    initializeInput(CRandom(0x123), clsPositions, 1, 2);
    static_cast<CSourceLayer*>(bert.GetLayer("cls_positions").Ptr())->SetBlob(clsPositions);

    std::unique_ptr<IPerformanceCounters> counters(MathEngine().CreatePerformanceCounters());
    counters->Synchronise();

    //for( int i = 0; i < numOfMeasures; ++i)
    CDnnUniformInitializer initializer(CRandom(0x123), 1, 2);
    bert.SetInitializer(&initializer);
    bert.RunOnce();

    auto output = static_cast<CSinkLayer*>(bert.GetLayer("output").Ptr())->GetBlob()->GetData();
    auto size = static_cast<CSinkLayer*>(bert.GetLayer("output").Ptr())->GetBlob()->GetDataSize();
    for (int i = 0; i < size; ++i) {
        printf("%d ", output.GetValueAt(i));
    }

    counters->Synchronise();
    std::cerr
        << '\n' << "Time: " << (double((*counters)[0].Value) / 1000000) / numOfMeasures << " ms."
        << '\t' << "Peak.Mem: " << (double(MathEngine().GetPeakMemoryUsage()) / 1024 / 1024) << " MB \n";
}

TEST(ThreadTest, BertThreadTest)
{
    std::unique_ptr<IPerformanceCounters> counters(MathEngine().CreatePerformanceCounters());
    counters->Synchronise();

    CRandom random(0x123);
    CDnn* bert = new CDnn(random, MathEngine());
    auto t = FileSystem::GetCurrentDir();
    CArchiveFile file(".\\RobertaTextSegmentationTrainNoLora.dnn", CArchive::load);
    CArchive archive(&file, CArchive::SD_Loading);

    bert->Serialize(archive);

    CArray<CDnn*> dnns;
    CObjectArray<CDnnBlob> input_ids;
    CObjectArray<CDnnBlob> clsPositions;
    CArray<TaskParams> taskParams;

    CPtr<CDnnBlob> input_ids0 = CDnnBlob::CreateTensor(MathEngine(), CT_Int, { 1, batch_size / numOfThreads, 512, 1, 1, 1, 1 });
    initializeInput(CRandom(0x123), input_ids0, 0, 50000);
    static_cast<CSourceLayer*>(bert->GetLayer("input_ids").Ptr())->SetBlob(input_ids0);


    CPtr<CDnnBlob> clsPositions0 = CDnnBlob::CreateTensor(MathEngine(), CT_Int, { 1, 1, 1, 1, 1, 1, paragraphs_cnt });
    initializeInput(CRandom(0x123), clsPositions0, 0, 512);
    static_cast<CSourceLayer*>(bert->GetLayer("cls_positions").Ptr())->SetBlob(clsPositions0);

    taskParams.Add({ bert });
    CArray<CRandom> randoms;

    for (int i = 0; i < numOfThreads - 1; ++i) {
        randoms.Add(CRandom(0x123));
        dnns.Add(bert->CreateReferenceDnn(randoms[i]));
        input_ids.Add(CDnnBlob::CreateTensor(MathEngine(), CT_Int, { 1, batch_size / numOfThreads, 512, 1, 1, 1, 1 }));
        clsPositions.Add(CDnnBlob::CreateTensor(MathEngine(), CT_Int, { 1, 1, 1, 1, 1, 1, paragraphs_cnt }));

        taskParams.Add({ dnns[i] });

        initializeInput(CRandom(0x123), input_ids[i], 0, 50000);
        initializeInput(CRandom(0x123), clsPositions[i], 0, 512);
        //MathEngine().VectorFill(clsPositions[i]->GetData<int>(), 2, clsPositions[i]->GetDataSize());
        //MathEngine().VectorFill(input_ids[i]->GetData<int>(), 2, input_ids[i]->GetDataSize());

        static_cast<CSourceLayer*>(dnns[i]->GetLayer("cls_positions").Ptr())->SetBlob(clsPositions[i]);
        static_cast<CSourceLayer*>(dnns[i]->GetLayer("input_ids").Ptr())->SetBlob(input_ids[i]);
    }

    counters->Synchronise();
    /*std::cerr
        << '\n' << "Time: " << (double((*counters)[0].Value) / 1000000) << " ms."
        << '\t' << "Peak.Mem: " << (double(MathEngine().GetPeakMemoryUsage()) / 1024 / 1024) << " MB \n";*/

    IThreadPool* pool = CreateThreadPool(numOfThreads);

   /* for (int i = 0; i < numOfThreads; ++i) {
        pool->AddTask(i, RunDnn, &(taskParams[i]));
    }

    pool->WaitAllTask();*/

    counters->Synchronise();

    for (int j = 0; j < numOfMeasures; ++j) {
        for (int i = 0; i < numOfThreads; ++i) {
            pool->AddTask(i, RunDnn, &(taskParams[i]));
        }
        pool->WaitAllTask();
    }

    counters->Synchronise();
    std::cerr
        << '\n' << "Time: " << (double((*counters)[0].Value) / 1000000) / numOfMeasures << " ms."
        << '\t' << "Peak.Mem: " << (double(MathEngine().GetPeakMemoryUsage()) / 1024 / 1024) << " MB \n";

    /*for (int i = 0; i < numOfThreads - 1; ++i) {
        EXPECT_TRUE( CompareBlobs( *( static_cast<CSinkLayer*>(bert->GetLayer("output").Ptr())->GetBlob() ),
            *( static_cast<CSinkLayer*>(dnns[i]->GetLayer("output").Ptr())->GetBlob() ), 1e-5f ) );
    }*/

    for (int i = 0; i < numOfThreads - 1; ++i) {
        delete dnns[i];
    }
    delete bert;
}

TEST(ThreadTest, DummyBertThread)
{
    CRandom random(0x123);
    CArray<CDnn*> dnns;
    CObjectArray<CDnnBlob> input_ids;
    CObjectArray<CDnnBlob> clsPositions;
    CArray<TaskParams> taskParams;


    for (int i = 0; i < numOfThreads; ++i) {
        CDnn* dnn = new CDnn(random, MathEngine());
        CArchiveFile file(".\\RobertaTextSegmentationTrainNoLora.dnn", CArchive::load);
        CArchive archive(&file, CArchive::SD_Loading);
        dnn->Serialize(archive);
        dnns.Add(dnn);
        input_ids.Add(CDnnBlob::CreateTensor(MathEngine(), CT_Int, { 1, batch_size / numOfThreads, 512, 1, 1, 1, 1 }));
        clsPositions.Add(CDnnBlob::CreateTensor(MathEngine(), CT_Int, { 1, 1, 1, 1, 1, 1, paragraphs_cnt }));

        taskParams.Add({ dnns[i] });

        initializeInput(CRandom(0x123), clsPositions[i], 0, 512);
        initializeInput(CRandom(0x123), input_ids[i], 0, 50000);
        /*MathEngine().VectorFill(clsPositions[i]->GetData<int>(), 2, clsPositions[i]->GetDataSize());
        MathEngine().VectorFill(input_ids[i]->GetData<int>(), 2, input_ids[i]->GetDataSize());*/

        static_cast<CSourceLayer*>(dnns[i]->GetLayer("cls_positions").Ptr())->SetBlob(clsPositions[i]);
        static_cast<CSourceLayer*>(dnns[i]->GetLayer("input_ids").Ptr())->SetBlob(input_ids[i]);
    }

    std::unique_ptr<IPerformanceCounters> counters(MathEngine().CreatePerformanceCounters());
    counters->Synchronise();
    IThreadPool* pool = CreateThreadPool(numOfThreads);

    for (int j = 0; j < numOfMeasures; ++j) {
        for (int i = 0; i < numOfThreads; ++i) {
            pool->AddTask(i, RunDnn, &(taskParams[i]));
        }
        pool->WaitAllTask();
    }

    counters->Synchronise();
    std::cerr
        << '\n' << "Time: " << (double((*counters)[0].Value) / 1000000) / numOfMeasures << " ms."
        << '\t' << "Peak.Mem: " << (double(MathEngine().GetPeakMemoryUsage()) / 1024 / 1024) << " MB \n";

   /* for (int i = 0; i < numOfThreads - 1; ++i) {
        EXPECT_TRUE(CompareBlobs(*(static_cast<CSinkLayer*>(dnns[i]->GetLayer("output").Ptr())->GetBlob()),
            *(static_cast<CSinkLayer*>(dnns[i + 1]->GetLayer("output").Ptr())->GetBlob()), 1e-3f));
    }*/

    for (int i = 0; i < numOfThreads; ++i) {
        delete dnns[i];
    }
}