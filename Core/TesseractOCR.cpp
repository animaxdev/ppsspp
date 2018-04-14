#include "Core/TesseractOCR.h"

#include <algorithm>

#include "baseapi.h"
#include "strngs.h"
#include "genericvector.h"
#include "allheaders.h"
#include "arrayaccess.h"
#include "params.h"


#include "file/vfs.h"
#include "net/url.h"
#include "net/http_client.h"
#include "base/logging.h"
#include "Core/Screenshot.h"
#include "Core/System.h"
#include "GPU/GPU.h"
#include "GPU/Common/GPUDebugInterface.h"
//#include "base/colorutil.h"
#include "UI/OnScreenDisplay.h"



bool FileReader(const STRING& filename, GenericVector<char>* data)
{
	int length = filename.length();
	const char * name = filename.c_str();
	if (length > 2 && name[0] == '.' && name[1] == '/')
	{
		name += 2;
	}
	ILOG("PSP_FileReader filename: %s", name);
	size_t size = 0;
	uint8_t *buffer = VFSReadFile(name, &size);
	if (buffer)
	{
		data->resize_no_init(size);
		memcpy(&(*data)[0], buffer, size);
		free(buffer);
		return true;
	}
	return false;
}



class TesseractOCR
{
public:
	TesseractOCR(const char* language);
	~TesseractOCR();

	bool IsRunning();
	void SetScaleFactor(float scale);
	void ProcessImage(const unsigned char * imagedata, int width, int height);
	void SetOutputPath(const char * path);
	
	void ProcessImageComponent(const unsigned char * imagedata, int width, int height);
	void ProcessComponentText(int x, int y, int width, int height);
	PIX* CreatePix(const unsigned char* imagedata, int width, int height);

	void TranslateText(const char * text);

	void DumpImg(PIXA *pixa, NUMA *na, const char * name);
	void DumpImg(PIXA *pixa, const char * name);
	void DumpImg(PIX *pix, const char * name, int index = -1);

protected:
	bool initTesseract();

	void NUMA_Process(PIX * pix, bool isMask);
	PIX * Preprocess(PIX * pix, bool postRemove);

	static void RunTesseractThread(TesseractOCR * tess);
	static void RunTextThread(TesseractOCR * tess);
	static void RunComponentThread(TesseractOCR * tess);

private:
	std::string language_;
	std::string outputPath_;
	tesseract::TessBaseAPI *baseAPI_;
	PIX * inputImage_;
	BOX * componentBox_;
	http::Client client_;
	float scaleFactor_;
	bool isRunning_;
};


TesseractOCR::TesseractOCR(const char* language) 
	: language_(language), baseAPI_(nullptr), scaleFactor_(4.0f), isRunning_(false)
{
}


TesseractOCR::~TesseractOCR()
{
	if (componentBox_ != nullptr)
	{
		boxDestroy(&componentBox_);
	}

	if (baseAPI_ != nullptr)
	{
		baseAPI_->End();
		delete baseAPI_;
		baseAPI_ = nullptr;
	}
}


bool TesseractOCR::IsRunning()
{
	return isRunning_;
}

void TesseractOCR::SetScaleFactor(float scale)
{
	scaleFactor_ = scale;
}

void TesseractOCR::SetOutputPath(const char * path)
{
	outputPath_.assign(path);
	if (!outputPath_.empty() && outputPath_.back() != '/')
	{
		outputPath_.push_back('/');
	}
}

bool TesseractOCR::initTesseract()
{
	if (baseAPI_ == nullptr)
	{
		ILOG("initialize tesseract");
		baseAPI_ = new tesseract::TessBaseAPI();
		if (baseAPI_->Init(nullptr, 0, language_.c_str(), tesseract::OEM_DEFAULT, nullptr, 0, nullptr, nullptr, false, FileReader))
		{
			ILOG("Could not initialize tesseract");
			delete baseAPI_;
			baseAPI_ = nullptr;
		}
		else
		{
			baseAPI_->SetPageSegMode(tesseract::PSM_SPARSE_TEXT);
		}

		if (baseAPI_ != nullptr)
		{
			bool resolved = client_.Resolve("fanyi.youdao.com", 80);
			if (resolved && !client_.Connect()) {
				ELOG("Failed connecting to server.");
				resolved = false;
			}
		}
	}

	return baseAPI_ == nullptr;
}


void TesseractOCR::ProcessImage(const unsigned char * imagedata, int width, int height)
{
	if (!isRunning_)
	{
		isRunning_ = true;
		PIX * image = CreatePix(imagedata, width, height);
		std::thread(&RunTesseractThread, image).detach();
	}
}

void TesseractOCR::NUMA_Process(PIX * pix, bool isMask)
{
	// Generate the pixa of 8 cc pieces.
	PIXA *pixa = nullptr;
	BOXA *boxa = pixConnComp(pix, &pixa, 8);

	// Extract the data we need about each component.
	NUMA *naw = nullptr;
	NUMA *nah = nullptr;
	pixaFindDimensions(pixa, &naw, &nah);
	//NUMA *nas = pixaCountPixels(pixa);



	// Build the indicator arrays for the set of components,
	// based on thresholds and selection criteria.
	NUMA *na1 = numaMakeThresholdIndicator(nah, isMask ? 6 : 4, L_SELECT_IF_GT);
	NUMA *na2 = numaMakeThresholdIndicator(naw, isMask ? 6 : 4, L_SELECT_IF_GT);
	NUMA *na3 = numaMakeThresholdIndicator(nah, isMask ? 140 : 90, L_SELECT_IF_LT);
	NUMA *na4 = numaMakeThresholdIndicator(naw, isMask ? 420 : 120, L_SELECT_IF_LT);

	// Combine the indicator arrays logically to find
	// the components that will be retained.
	NUMA *nad = numaLogicalOp(NULL, na1, na2, L_UNION);
	numaLogicalOp(nad, nad, na3, L_INTERSECTION);
	numaLogicalOp(nad, nad, na4, L_INTERSECTION);

	if (!isMask)
	{
		NUMA *nar = pixaFindPerimSizeRatio(pixa);
		NUMA *na5 = numaMakeThresholdIndicator(nar, 0.7f, L_SELECT_IF_GT);
		numaLogicalOp(nad, nad, na5, L_INTERSECTION);

		numaDestroy(&nar);
		numaDestroy(&na5);
	}


	// Invert to get the components that will be removed.
	numaInvert(nad, nad);


	// Remove the components, in-place.
	pixRemoveWithIndicator(pix, pixa, nad);


	numaDestroy(&na1);
	numaDestroy(&na2);
	numaDestroy(&na3);
	numaDestroy(&na4);

	numaDestroy(&naw);
	numaDestroy(&nah);

	boxaDestroy(&boxa);
	pixaDestroy(&pixa);
}

void TesseractOCR::ProcessImageComponent(const unsigned char * imagedata, int width, int height)
{
	if (!isRunning_)
	{
		isRunning_ = true;
		PIX * image = CreatePix(imagedata, width, height);
		std::thread(&RunComponentThread, image).detach();
	}
}

void TesseractOCR::ProcessComponentText(int x, int y, int width, int height)
{
	if (baseAPI_ != nullptr && !isRunning_)
	{
		isRunning_ = true;
		BOX * box = boxCreate(x * scaleFactor_, y * scaleFactor_, width * scaleFactor_, height * scaleFactor_);
		std::thread(&RunTextThread, box).detach();
	}
}


PIX* TesseractOCR::CreatePix(const unsigned char* imagedata, int width, int height)
{
	int bytes_per_pixel = 3;
	int bytes_per_line = width * bytes_per_pixel;
	int bpp = bytes_per_pixel * 8;
	if (bpp == 0) bpp = 1;
	PIX* pix = pixCreate(width, height, bpp == 24 ? 32 : bpp);
	l_uint32* data = pixGetData(pix);

	// Put the colors in the correct places in the line buffer.
	for (int y = 0; y < height; ++y, imagedata += bytes_per_line)
	{
		for (int x = 0; x < width; ++x, ++data)
		{
			SET_DATA_BYTE(data, COLOR_RED, imagedata[3 * x]);
			SET_DATA_BYTE(data, COLOR_GREEN, imagedata[3 * x + 1]);
			SET_DATA_BYTE(data, COLOR_BLUE, imagedata[3 * x + 2]);
		}
	}

	pixSetYRes(pix, 300);
	return pix;
}


void TesseractOCR::TranslateText(const char * text)
{
	UrlEncoder encoder;
	encoder.Add("keyfrom", "longcwang");
	encoder.Add("key", "131895274");
	encoder.Add("type", "data");
	encoder.Add("doctype", "json");
	encoder.Add("version", "1.1");
	encoder.Add("q", text);

	std::string api("http://fanyi.youdao.com/openapi.do?");
	Url fanyiUrl(api + encoder.ToString());
	Buffer output;
	float progress = 0.0f;
	bool cancelled = false;

	int result = client_.GET(fanyiUrl.Resource().c_str(), &output, &progress, &cancelled);
	if (result == 200)
	{
		std::string response;
		output.PeekAll(&response);
		ILOG("zhangwei TakeGameAction client result: %d, %s", result, response.c_str());
	}
	else
	{
		ILOG("zhangwei TakeGameAction client result: %d", result);
	}
}



void TesseractOCR::RunTesseractThread(TesseractOCR * tess)
{
	if (!tess->initTesseract()) {
		pixDestroy(&tess->inputImage_);
		return;
	}

	tess->DumpImg(tess->inputImage_, "screenshot");
	PIX * test = tess->Preprocess(tess->inputImage_, false);
	pixDestroy(&tess->inputImage_);

	tess->baseAPI_->SetImage(test);
	pixDestroy(&test);

	Pixa* croppedPixa;
	Boxa* boxes = tess->baseAPI_->GetComponentImages(tesseract::RIL_TEXTLINE, true, &croppedPixa, NULL);
	ILOG("Found image components: %d", boxes->n);
	for (int i = 0; i < boxes->n; ++i)
	{
		BOX* box = boxaGetBox(boxes, i, L_CLONE);

		//PIX* cropped = pixClipRectangle(TessBaseAPI->GetInputImage(), box, NULL);
		//DebugImg(croppedPixa, "cropped");

		ILOG("Box[%d]: x=%d, y=%d, w=%d, h=%d", i, box->x, box->y, box->w, box->h);
		if (box->h > 20 && box->h < 100)
		{
			tess->baseAPI_->SetRectangle(box->x, box->y, box->w, box->h);
			char* ocrResult = tess->baseAPI_->GetUTF8Text();
			int conf = tess->baseAPI_->MeanTextConf();

			ILOG("Conf[%d]: %d", i, conf);
			if (conf > 30)
			{
				std::string str(ocrResult);
				str.erase(remove_if(str.begin(), str.end(), isspace), str.end());
				ILOG("result: %s", str.c_str());
				osm.ShowXY(str.c_str(), box->x / 2, box->y / 2);
			}
			delete[] ocrResult;
		}
		boxDestroy(&box);
	}
	pixaDestroy(&croppedPixa);
	boxaDestroy(&boxes);
	//client.Disconnect();
	tess->isRunning_ = false;
	ILOG("zhangwei RunTesseractThread end");
}



void TesseractOCR::RunComponentThread(TesseractOCR * tess)
{
	if (!tess->initTesseract()) {
		pixDestroy(&tess->inputImage_);
		return;
	}

	PIX * test = tess->Preprocess(tess->inputImage_, false);
	pixDestroy(&tess->inputImage_);

	tess->baseAPI_->SetImage(test);
	pixDestroy(&test);

	float scaleFactor = tess->scaleFactor_;
	Boxa* boxes = tess->baseAPI_->GetComponentImages(tesseract::RIL_TEXTLINE, true, nullptr, nullptr);
	ILOG("Found image components: %d", boxes->n);
	for (int i = 0; i < boxes->n; ++i)
	{
		BOX* box = boxaGetBox(boxes, i, L_CLONE);
		//ILOG("Box[%d]: x=%.02f, y=%.02f, w=%.02f, h=%.02f", i, box->x / scaleFactor, box->y / scaleFactor, box->w / scaleFactor, box->h / scaleFactor);
		osm.ShowRect(box->x / scaleFactor, box->y / scaleFactor, box->w / scaleFactor, box->h / scaleFactor, 0xFF0000);
		boxDestroy(&box);
	}
	boxaDestroy(&boxes);
	tess->isRunning_ = false;
	ILOG("zhangwei RunTesseractThread end");
}

void TesseractOCR::RunTextThread(TesseractOCR * tess)
{
	BOX* box = tess->componentBox_;
	tess->baseAPI_->SetRectangle(box->x, box->y, box->w, box->h);

	char* ocrResult = tess->baseAPI_->GetUTF8Text();
	int conf = tess->baseAPI_->MeanTextConf();

	ILOG("MeanTextConf: %d", conf);
	float scaleFactor = tess->scaleFactor_;
	std::string str(ocrResult);
	str.erase(remove_if(str.begin(), str.end(), isspace), str.end());
	ILOG("result: %s", str.c_str());
	osm.ShowXY(str.c_str(), box->x / scaleFactor, box->y / scaleFactor);

	delete[] ocrResult;
	tess->isRunning_ = false;
	ILOG("zhangwei RunTesseractThread end");
}


PIX * TesseractOCR::Preprocess(PIX * pix, bool postRemove)
{
	int i = 0;
	PIX * output = pix;

	// 8比特灰度图
	PIX * gray = pixConvertRGBToGray(output, 0.0f, 0.0f, 0.0f);
	output = gray;
	//DebugImg(output, "0_gray", i++);

	// 灰度图放大
	PIX *scaled_pix = pixScaleGrayLI(output, 2, 2);
	pixDestroy(&output);
	output = scaled_pix;
	//DebugImg(output, "0_scale", i++);

	// 字体是白色，需要反转颜色成黑色，因为识别使用黑色
	pixInvert(output, output);
	//DebugImg(output, "0_invert", i++);


	// 为图案模糊描边
	PIX * erode = pixErodeGray(output, 5, 5);
	pixThresholdToValue(erode, erode, 220, 255);
	//DebugImg(erode, "0_erode", i++);

	// 黑白图
	PIX * mask = pixThresholdToBinary(erode, 130);
	//DebugImg(mask, "0_mask1", i++);

	NUMA_Process(mask, true);
	//DebugImg(mask, "0_mask2", i++);

	// 白色底板
	pixSetAll(erode);

	// 根据mask，将output的内容复制到白色底板上
	pixCombineMasked(erode, output, mask);
	pixDestroy(&mask);
	pixDestroy(&output);
	output = erode;


	if (postRemove)
	{
		PIX * binary = pixThresholdToBinary(output, 130);
		pixDestroy(&output);
		output = binary;

		NUMA_Process(output, false);
	}

	DumpImg(output, "0_output", i++);

	return output;
}

void TesseractOCR::DumpImg(PIXA *pixa, NUMA *na, const char * name)
{
	if (pixa == nullptr || na == nullptr)
	{
		ILOG("zhangwei debugImg pixa, na is null");
		return;
	}

	char buffer[128];
	int n = pixaGetCount(pixa);
	for (int i = 0; i < n; ++i)
	{
		float value = 0;
		PIX* cropped = pixaGetPix(pixa, i, L_CLONE);
		numaGetFValue(na, i, &value);
		sprintf(buffer, "%s%s_%03d_%03.03f.png", outputPath_.c_str(), name, i, value);
		pixWrite(buffer, cropped, IFF_PNG);
		pixDestroy(&cropped);
	}
}


void TesseractOCR::DumpImg(PIXA *pixa, const char * name)
{
	if (pixa == nullptr)
	{
		ILOG("zhangwei debugImg pixa is null");
		return;
	}

	char buffer[128];
	int n = pixaGetCount(pixa);
	for (int i = 0; i < n; ++i)
	{
		PIX* cropped = pixaGetPix(pixa, i, L_CLONE);
		sprintf(buffer, "%s%s_%03d.png", outputPath_.c_str(), name, i);
		pixWrite(buffer, cropped, IFF_PNG);
		pixDestroy(&cropped);
	}
}


void TesseractOCR::DumpImg(PIX *pix, const char * name, int index = -1)
{
	if (pix == nullptr)
	{
		ILOG("zhangwei debugImg pix is null");
		return;
	}

	char buffer[128];
	if (index != -1)
	{
		sprintf(buffer, "%s%s_%03d.png", outputPath_.c_str(), name, index);
	}
	else
	{
		sprintf(buffer, "%s%s.png", outputPath_.c_str(), name);
	}
	pixWrite(buffer, pix, IFF_PNG);
}















TesseractOCR Tesseract;
bool TessActionImage()
{
	if (!gpuDebug || Tesseract.IsRunning())
	{
		ERROR_LOG(SYSTEM, "Can't take screenshots when GPU not running");
		return false;
	}

	bool success = false;
	GPUDebugBuffer buf;
	if (gpuDebug->GetCurrentFramebuffer(buf, GPU_DBG_FRAMEBUF_DISPLAY, -1))
	{
		u8 *flipbuffer = nullptr;
		u32 width = PSP_CoreParameter().renderWidth;
		u32 height = PSP_CoreParameter().renderHeight;
		const u8 *buffer = ConvertBufferToScreenshot(buf, false, flipbuffer, width, height);
		Tesseract.ProcessImageComponent(buffer, width, height);
		delete[] flipbuffer;
		success = true;
	}
	return success;
}


bool TessActionText(int x, int y, int width, int height)
{
	if (!gpuDebug || Tesseract.IsRunning())
	{
		return false;
	}
	Tesseract.ProcessComponentText(x, y, width, height);
	return true;
}



PPSSPPOCR::PPSSPPOCR() {

}

PPSSPPOCR::~PPSSPPOCR() {

}

void PPSSPPOCR::MarkAllTexts() {

}

void PPSSPPOCR::HideAllMarks() {

}

void PPSSPPOCR::ProcessText(int x, int y, int width, int height) {

}

void PPSSPPOCR::TranslateText(const char * text) {

}

void PPSSPPOCR::ShowProgressSpinner() {

}

void PPSSPPOCR::HideProgressSpinner() {

}
