/* Copyright © 2017-2023 ABBYY

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

#include <NeoMathEngine/ThreadPool.h>
#include <NeoMathEngine/NeoMathEngineException.h>

#include <condition_variable>
#include <mutex>
#include <thread>
#include <queue>
#include <cstdint>

#if FINE_PLATFORM( FINE_LINUX )
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif // _GNU_SOURCE
#include <fstream>
#include <pthread.h>
#include <sched.h>
#include <sys/stat.h>
#endif // FINE_PLATFORM( FINE_LINUX )

namespace NeoML {

#if FINE_PLATFORM( FINE_LINUX )
// Checks if we're running inside of docker or k8s
static bool isInDocker()
{
	auto impl = [] () -> bool
	{
		// First method: check the existence of .dockerenv
		struct stat buffer;
		if( ::stat( "/.dockerenv", &buffer ) == 0 ) {
			return true;
		}

		// Second method: checking the contents of cgroup file
		std::ifstream cgroupFile( "/proc/self/cgroup" );
		if( cgroupFile.good() ) {
			std::string data;
			while( cgroupFile >> data ) {
				if( data.find( "docker" ) != std::string::npos || data.find( "kubepods" ) != std::string::npos ) {
					return true;
				}
			}
		}

		return false;
	};

	static const bool isInDocker = impl();
	return isInDocker;
}

// Reads one value from file
// Returns default if something goes wrong
template<class T>
static T readValueFromFile( const char* name, const T defaultValue )
{
	std::ifstream stream( name );
	T result = defaultValue;
	if( stream.good() && ( stream >> result ) ) {
		return result;
	}
	return defaultValue;
}
#endif // FINE_PLATFORM( FINE_LINUX )

int GetAvailableCpuCores()
{
	auto impl = [] () -> int
	{
	#if FINE_PLATFORM( FINE_LINUX )
		if( isInDocker() ) {
			// Case #1: linux Docker with --cpus value set (or k8s with cpu limits)
			// When working under cgroups without quotas cfs_quota_us contains -1
			const int quota = readValueFromFile<int>( "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", -1 );
			const int period = readValueFromFile<int>( "/sys/fs/cgroup/cpu/cpu.cfs_period_us", -1 );
			if( quota > 0 && period > 0 ) {
				// Using ceil because --cpus 0.1 is a valid scenario in docker (0.1 means quota * 10 == period)
				return ( quota + period - 1 ) / period;
			}

			// Case #2: linux Docker with --cpuset-cpus
			cpu_set_t cpuSet;
			CPU_ZERO( &cpuSet );
			if( ::pthread_getaffinity_np( ::pthread_self(), sizeof( cpu_set_t ), &cpuSet ) == 0 ) {
				return static_cast<int>( CPU_COUNT( &cpuSet ) );
			}
		}
	#endif // FINE_PLATFORM( FINE_LINUX )
		// std::thread::hardware_concurrency may return 0 if the value is not well defined or not computable
		return std::max( static_cast<int>( std::thread::hardware_concurrency() ), 1 );
	};

	static const int availabeCpuCores = impl();
	return availabeCpuCores;
}

size_t GetRamLimit()
{
#if FINE_PLATFORM( FINE_LINUX )
	if( isInDocker() ) {
		const uint64_t memLimit = readValueFromFile<uint64_t>( "/sys/fs/cgroup/memory/memory.limit_in_bytes", 0 );
		const uint64_t memUsed = readValueFromFile<uint64_t>( "/sys/fs/cgroup/memory/memory.usage_in_bytes", 0 );
		if( memLimit > memUsed ) {
			return static_cast<size_t>( std::min( SIZE_MAX, memLimit - memUsed ) );
		}
	}
#endif // FINE_PLATFORM( FINE_LINUX )

	return SIZE_MAX; // no limit
}

// Returns memory limit 

//------------------------------------------------------------------------------------------------------------

class CThreadPoolEmpty : public IThreadPool {
public:
	CThreadPoolEmpty() = default;
	// IThreadPool:
	int Size() const override { return 1; }
	bool AddTask( int, TFunction, void* ) override { ASSERT_EXPR( false ); return false; }
	void WaitAllTask() override { ASSERT_EXPR( false ); }
};

//------------------------------------------------------------------------------------------------------------

class CThreadPool : public IThreadPool {
public:
	explicit CThreadPool( int threadCount );
	~CThreadPool() override;

	// IThreadPool:
	int Size() const override { return static_cast<int>( threads.size() ); }
	bool AddTask( int threadIndex, TFunction function, void* params ) override;
	void WaitAllTask() override;

private:
	struct CTask final {
		IThreadPool::TFunction Function{};
		void* Params{};
	};

	struct CParams final {
		int Count{}; // Number of threads in the pool.
		int Index{}; // Thread index in the pool.
		std::condition_variable ConditionVariable{};
		std::mutex Mutex{};
		std::queue<CTask> Queue{};
		bool Stopped{};
	};

	static void threadEntry( CParams* );
	// Stops all threads and waits for them to complete.
	void stopAndWait();

	std::vector<std::thread*> threads{}; // CPointerArray isn't available in neoml.
	std::vector<CParams*> params{};
};

void CThreadPool::threadEntry( CParams* parameters )
{
	CParams& params = *parameters;
	std::unique_lock<std::mutex> lock( params.Mutex );

	while( !params.Stopped ) {
		if( !params.Queue.empty() ) {
			CTask task = params.Queue.front();
			lock.unlock();

			try {
				task.Function( params.Index, task.Params );
			} catch( ... ) {
				ASSERT_EXPR( false ); // Better than nothing
			}

			lock.lock();
			params.Queue.pop();
			params.ConditionVariable.notify_all();
		}

		params.ConditionVariable.wait( lock );
	}
}

CThreadPool::CThreadPool( int threadCount )
{
	ASSERT_EXPR( threadCount > 0 );
	for( int i = 0; i < threadCount; i++ ) {
		CParams* param = new CParams();
		param->Count = threadCount;
		param->Index = i;
		param->Stopped = false;
		params.push_back( param );

		std::thread* thread = new std::thread( threadEntry, param );
		threads.push_back( thread );
	}
}

CThreadPool::~CThreadPool()
{
	stopAndWait();
	for( auto t : threads ) {
		delete t;
	}
	for( auto p : params ) {
		delete p;
	}
}

bool CThreadPool::AddTask( int threadIndex, TFunction function, void* functionParams )
{
	assert( 0 <= threadIndex && threadIndex < Size() );

	std::unique_lock<std::mutex> lock( params[threadIndex]->Mutex );
	params[threadIndex]->Queue.push( { function, functionParams } );
	params[threadIndex]->ConditionVariable.notify_all();

	return !params[threadIndex]->Stopped;
}

void CThreadPool::WaitAllTask()
{
	for( size_t i = 0; i < params.size(); i++ ) {
		std::unique_lock<std::mutex> lock( params[i]->Mutex );
		while( !params[i]->Queue.empty() ) {
			params[i]->ConditionVariable.wait( lock );
		}
	}
}

void CThreadPool::stopAndWait()
{
	for( size_t i = 0; i < threads.size(); i++ ) {
		{
			std::unique_lock<std::mutex> lock( params[i]->Mutex );
			params[i]->Stopped = true;
		}
		params[i]->ConditionVariable.notify_all();
		threads[i]->join();
	}
}

//------------------------------------------------------------------------------------------------------------

IThreadPool* CreateThreadPool( int threadCount )
{
	if( threadCount <= 0 ) {
		threadCount = GetAvailableCpuCores();
	}
	if( threadCount == 1 ) {
		return new CThreadPoolEmpty();
	}
	return new CThreadPool( threadCount );
	// TODO: Add here creation of any other implementations of ThreadPool
}

} // namespace NeoML
