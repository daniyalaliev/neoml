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
#include <mutex>
#pragma hdrstop

#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

namespace NeoMLTest {

struct CDnnTestParam {
	CDnn* net;
};

static IThreadPool::TFunction runDnn = [](int, void* params)
{
	CDnnTestParam* taskParams = static_cast<CDnnTestParam*>(params);
	taskParams->net->RunOnce();
};

static CDnn* createDnn(CRandom& random)
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

} // namespace NeoMLTest

//------------------------------------------------------------------------------------------------

TEST(ReferenceDnnTest, ReferenceDnnsThreads)
{
	const int numOfThreads = 4;

	CObjectArray<CDnnBlob> blobs;
	CArray<CDnnTestParam> taskParams;
	CArray<CRandom> randoms;
	CArray<CDnn*> dnns;

	IPerformanceCounters* counters(MathEngine().CreatePerformanceCounters());
	counters->Synchronise();

	for(int i = 0; i < numOfThreads; ++i) {
		randoms.Add(CRandom(i));
		if(i == 0) {
			dnns.Add(createDnn(randoms[i]));
		} else {
			dnns.Add(dnns[0]->CreateReferenceDnn(randoms[i]));
		}
		blobs.Add(CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 8, 20, 30, 10 }));
		taskParams.Add({ dnns[i] });
		
		CRandom randomInit(0);
		for(int j = 0; j < blobs[i]->GetDataSize(); ++j) {
			blobs[i]->GetData().SetValueAt(j, static_cast<float>(randomInit.Uniform(-1, 1)));
		}
		static_cast<CSourceLayer*>(dnns[i]->GetLayer("in").Ptr())->SetBlob(blobs[i]);
	}

	IThreadPool* pool = CreateThreadPool(numOfThreads);
	for(int i = 0; i < numOfThreads; ++i) {
		pool->AddTask(i, runDnn, &(taskParams[i]));
	}

	pool->WaitAllTask();
	counters->Synchronise();
	std::cerr
		<< '\n' << "Time: " << (double((*counters)[0].Value) / 1000000) << " ms."
		<< '\t' << "Peak.Mem: " << (double(MathEngine().GetPeakMemoryUsage()) / 1024 / 1024) << " MB \n";

	for(int i = 0; i < numOfThreads - 1; ++i) {
		EXPECT_TRUE(CompareBlobs(
			*(static_cast<CSourceLayer*>(dnns[i]->GetLayer("in").Ptr())->GetBlob()),
			*(static_cast<CSourceLayer*>(dnns[i + 1]->GetLayer("in").Ptr())->GetBlob()))
		);

		EXPECT_TRUE(CompareBlobs(
			*(static_cast<CSinkLayer*>(dnns[i]->GetLayer("sink").Ptr())->GetBlob()),
			*(static_cast<CSinkLayer*>(dnns[i+1]->GetLayer("sink").Ptr())->GetBlob()))
		);
	}

	delete counters;
	delete pool;
	for(int i = 0; i < numOfThreads; ++i) {
		delete dnns[i];
	}
}

TEST(ReferenceDnnTest, FullCopyDnnsThreads)
{
	const int numOfThreads = 4;

	CArray<CDnn*> dnns;
	CObjectArray<CDnnBlob> blobs;
	CArray<CDnnTestParam> taskParams;
	CArray<CRandom> randoms;

	IPerformanceCounters* counters(MathEngine().CreatePerformanceCounters());
	counters->Synchronise();

	for(int i = 0; i < numOfThreads; ++i) {
		randoms.Add(CRandom(0x123));
		dnns.Add(createDnn(randoms[i]));
		blobs.Add(CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 8, 20, 30, 10 }));
		taskParams.Add({ dnns[i] });
		CRandom randomInit(0);

		for(int j = 0; j < blobs[i]->GetDataSize(); ++j) {
			blobs[i]->GetData().SetValueAt(j, static_cast<float>(randomInit.Uniform(-1, 1)));
		}

		static_cast<CSourceLayer*>(dnns[i]->GetLayer("in").Ptr())->SetBlob(blobs[i]);
	}

	IThreadPool* pool = CreateThreadPool(numOfThreads);
	for(int i = 0; i < numOfThreads; ++i) {
		pool->AddTask(i, runDnn, &(taskParams[i]));
	}

	pool->WaitAllTask();
	counters->Synchronise();
	std::cerr
		<< '\n' << "Time: " << (double((*counters)[0].Value) / 1000000) << " ms."
		<< '\t' << "Peak.Mem: " << (double(MathEngine().GetPeakMemoryUsage()) / 1024 / 1024) << " MB \n";

	delete counters;
	delete pool;
	for(int i = 0; i < numOfThreads; ++i) {
		delete dnns[i];
	}
}
