


class PPSSPPOCR
{
public:
	PPSSPPOCR();
	~PPSSPPOCR();

	void MarkAllTexts();
	void HideAllMarks();

	void ProcessText(int x, int y, int width, int height);

	void TranslateText(const char * text);

	void ShowProgressSpinner();
	void HideProgressSpinner();
};
