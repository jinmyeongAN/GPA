import re
import warnings
from io import BytesIO
from typing import List, Optional, Mapping, Any, Iterator

import numpy as np
import pdfplumber.page
from langchain.docstore.document import Document
from langchain.document_loaders import PDFPlumberLoader
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.parsers import PDFPlumberParser

_PDF_FILTER_WITH_LOSS = ["DCTDecode", "DCT", "JPXDecode"]
_PDF_FILTER_WITHOUT_LOSS = [
    "LZWDecode",
    "LZW",
    "FlateDecode",
    "Fl",
    "ASCII85Decode",
    "A85",
    "ASCIIHexDecode",
    "AHx",
    "RunLengthDecode",
    "RL",
    "CCITTFaxDecode",
    "CCF",
    "JBIG2Decode",
]


class PDFPlumberLoaderPlus(PDFPlumberLoader):
    def load(self) -> List[Document]:
        """Load file."""

        parser = PDFPlumberParserPlus(
            text_kwargs=self.text_kwargs,
            dedupe=self.dedupe,
            extract_images=self.extract_images,
        )
        blob = Blob.from_path(self.file_path)
        return parser.parse(blob)


class PDFPlumberParserPlus(PDFPlumberParser):
    """Parse `PDF` with `PDFPlumber`."""

    def __init__(self, text_kwargs: Optional[Mapping[str, Any]] = None, dedupe: bool = False,
                 extract_images: bool = False) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``pdfplumber.Page.extract_text()``
            dedupe: Avoiding the error of duplicate characters if `dedupe=True`.
        """
        super().__init__(text_kwargs, dedupe, extract_images)
        if self.extract_images:
            from pix2tex.cli import LatexOCR
            self.ocr_model = LatexOCR()

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pdfplumber

        with blob.as_bytes_io() as file_path:
            doc = pdfplumber.open(file_path)  # open document

            yield from [
                Document(
                    page_content=self._process_page_content(page),
                    metadata=dict(
                        {
                            "source": blob.source,
                            "file_path": blob.source,
                            "page": page.page_number - 1,
                            "total_pages": len(doc.pages),
                        },
                        **{
                            k: doc.metadata[k]
                            for k in doc.metadata
                            if type(doc.metadata[k]) in [str, int]
                        },
                    ),
                )
                for page in doc.pages
            ]

    def _wordlist_to_text(self, word_list, page_width, text_kwargs=None):
        if text_kwargs is None:
            text_kwargs = {}
        if 'gap_tolerance' not in text_kwargs:
            text_kwargs['gap_tolerance'] = 12
        if 'indent_tolerance' not in text_kwargs:
            text_kwargs['indent_tolerance'] = 27
        if 'bullet_point_chars' not in text_kwargs:
            text_kwargs['bullet_point_chars'] = '•◦‣⁃■□▪▫-○◘◙◎●◦◉◌◍◐◑◒◓◔◕◖◗◘◙◚◛◜◝◞◟◠◡◢◣◤◥◦◧◨◩◪◫◬◭◮◯◰◱◲◳◴◵◶◷◸◹◺◻◼◽◾◿'

        sentence_list = []
        indent_num = 0
        is_title = True
        bullet_point = False
        # indent_size = self.indent_tolerance
        indent_size = text_kwargs['indent_tolerance']
        font_size = word_list[0]['size']
        sent_x0 = word_list[0]['x0']  # for indent length determination
        word_x1 = word_list[0]['x1']  # for sentence determination
        if 'Bold' in word_list[0]['fontname']:
            is_bold = True
            sent_str = '*'
        else:
            is_bold = False
            sent_str = ''
        sent_str += word_list[0]['text']
        for word in word_list[1:]:
            if word['text'] in text_kwargs['bullet_point_chars']:
                bullet_point = True
                is_bold = False
                continue
            if ((word_x1 - text_kwargs['gap_tolerance'] <= word['x0'] <= word_x1 + text_kwargs['gap_tolerance']
                 or (word['x0'] - indent_size <= sent_x0 <= word['x0'] + indent_size
                     and word_x1 > page_width * 0.75
                     and word['size'] - 0.1 <= font_size <= word['size'] + 0.1))
                    and not bullet_point):  # if word is next part of current text
                if word['x0'] <= word_x1 + text_kwargs['gap_tolerance']:
                    sent_str += ' '
                if is_bold is False and 'Bold' in word['fontname']:
                    is_bold = True
                    sent_str += '*'
                if sent_str[-1] not in [' ', '\t'] and word['text'] not in [' ', '\t']:
                    sent_str = sent_str + ' ' + word['text']
                else:
                    sent_str = sent_str + word['text']
                if is_bold is True and 'Bold' not in word['fontname']:
                    is_bold = False
                    sent_str += '*'
                font_size = word['size']
                word_x1 = word['x1']
            else:  # if word is first word of next sentence
                if is_bold:
                    sent_str += '*'
                sentence_list.append(
                    re.sub(r' +', ' ', re.sub(r'( *\t+ *)+', '\t', sent_str.rstrip().replace('\xa0', ' '))))
                # sentence_list.append(re.sub(r'[ \t]+', '\t', sent_str.rstrip().replace('\xa0', ' ')))
                # sentence_list.append(sent_str.rstrip().replace('\xa0', ' '))
                if is_title:
                    indent_num = 0
                elif sent_x0 + indent_size <= word['x0'] or sent_x0 - indent_size >= word['x0']:
                    # if indent_size == self.indent_tolerance:
                    #     indent_size = abs(sent_x0 - word['x0']) * 0.95
                    indent_num += min(int((word['x0'] - sent_x0) / indent_size), 1)
                    indent_num = max(indent_num, 0)
                    """
                    if sent_x0 + indent_size <= word['x0']:
                        indent_num += int((word['x0'] - sent_x0) / indent_size)
                    elif sent_x0 - indent_size >= word['x0']:
                        indent_num = max(indent_num - 1, 0)
                    """
                sent_str = '\t' * indent_num
                if bullet_point:
                    sent_str += '- '
                    bullet_point = False
                is_title = False
                sent_x0 = word['x0']
                word_x1 = word['x1']
                font_size = word['size']
                if 'Bold' in word['fontname']:
                    is_bold = True
                    sent_str += '*'
                else:
                    is_bold = False
                sent_str += word['text']
        if sent_str:
            page_data = re.sub(r' +', ' ', re.sub(r'( *\t+ *)+', '\t', sent_str.rstrip().replace('\xa0', ' ')))
            # page_data = re.sub(r'[ \t]+', '\t', sent_str.rstrip().replace('\xa0', ' '))
            try:
                _ = int(page_data)  # if sentence is a number, assume it to page number
            except ValueError:
                sentence_list.append(page_data)
        return '\n'.join(sentence_list)

    def _process_page_content(self, page: pdfplumber.page.Page) -> str:
        """Process the page content based on dedupe."""
        if self.dedupe:
            word_list = page.dedupe_chars().extract_words(x_tolerance=3,
                                                          y_tolerance=10,
                                                          keep_blank_chars=True,
                                                          extra_attrs=['fontname', 'size', 'stroking_color',
                                                                       'non_stroking_color'])
        else:
            word_list = page.extract_words(x_tolerance=3,
                                           y_tolerance=10,
                                           keep_blank_chars=True,
                                           extra_attrs=['fontname', 'size', 'stroking_color', 'non_stroking_color'])

        image_word_list = self._convert_images_to_words_from_page(page)

        for image_word in image_word_list:
            for word_idx in range(len(word_list)):
                word_avg_y = (word_list[word_idx]['top'] + word_list[word_idx]['bottom']) / 2
                if (word_avg_y > image_word['top'] and word_list[word_idx]['x0'] > image_word['x1']) \
                        or word_avg_y > image_word['bottom']:
                    word_list.insert(word_idx, image_word)
                    break
            else:
                word_list.append(image_word)

        return self._wordlist_to_text(word_list, page.width)

    def _convert_images_to_words_from_page(self, page: pdfplumber.page.Page) -> list:
        """Extract images from page and get the text with LaTeX-OCR."""
        from PIL import Image, ImageChops

        if not self.extract_images:
            return []

        images = []
        for img in page.images:
            if img["stream"]["Filter"].name in _PDF_FILTER_WITHOUT_LOSS:
                images.append(
                    [Image.fromarray(
                        np.frombuffer(img["stream"].get_data(), dtype=np.uint8).reshape(img["stream"]["Height"],
                                                                                        img["stream"]["Width"],
                                                                                        -1)).convert('RGB'), img]
                )
            elif img["stream"]["Filter"].name in _PDF_FILTER_WITH_LOSS:
                images.append([Image.open(BytesIO(img["stream"].get_data())).convert('RGB'), img])
                # images.append([img["stream"].get_data(), img])
            else:
                warnings.warn("Unknown PDF Filter!")

        filted_images = []
        for image_tuple in images:
            # check if image is mono-color
            bg_color = image_tuple[0].getpixel((0, image_tuple[0].height - 1))
            pixels = image_tuple[0].load()
            bg_ratio = 0
            bg_horizontal_ratio = []
            for i in range(image_tuple[0].size[1]):  # for every pixel:
                bg_hori_ratio = 0
                for j in range(image_tuple[0].size[0]):
                    # if pixels[j, i] == bg_color:
                    if (bg_color[0] - 20 <= pixels[j, i][0] <= bg_color[0] + 20
                            and bg_color[1] - 20 <= pixels[j, i][1] <= bg_color[1] + 20
                            and bg_color[2] - 20 <= pixels[j, i][2] <= bg_color[2] + 20):
                        # change background to white
                        pixels[j, i] = (255, 255, 255)
                        bg_hori_ratio += 1
                        bg_ratio += 1
                bg_horizontal_ratio.append(bg_hori_ratio / image_tuple[0].size[0])

            # print(bg_ratio / (image_tuple[0].size[0] * image_tuple[0].size[1]))
            if bg_ratio / (image_tuple[0].size[0] * image_tuple[0].size[1]) < 0.7:
                # if background is not white enough, skip
                continue

            rgb = image_tuple[0].split()
            if ImageChops.difference(rgb[0], rgb[1]).getextrema()[1] > 30:
                continue
            if ImageChops.difference(rgb[0], rgb[2]).getextrema()[1] > 30:
                continue
            if ImageChops.difference(rgb[1], rgb[2]).getextrema()[1] > 30:
                continue
            filted_images.append(image_tuple + [bg_horizontal_ratio])

        image_dict_list = []
        for image_tuple in filted_images:
            bg_horizontal_ratio = image_tuple[2]
            # get every non-one (= background) segment and its pos
            segment_dict_list = []
            is_bg = True
            for i in range(len(bg_horizontal_ratio)):
                if is_bg and bg_horizontal_ratio[i] != 1:
                    segment_dict_list.append({'start': i})
                    is_bg = False
                elif not is_bg and bg_horizontal_ratio[i] == 1:
                    segment_dict_list[-1]['end'] = i
                    is_bg = True
            if not is_bg:
                segment_dict_list[-1]['end'] = len(bg_horizontal_ratio)
            # get segment height
            segment_dict_list_temp = []
            line_num = 0
            for segment_dict in segment_dict_list:
                height = segment_dict['end'] - segment_dict['start']
                if height <= 10:
                    continue
                bg_ratio = sum(bg_horizontal_ratio[segment_dict['start']:segment_dict['end']]) / height
                if bg_ratio <= 0.97:
                    line_num += 1
                segment_dict_list_temp.append({'start': segment_dict['start'],
                                               'end': segment_dict['end'],
                                               'bg_ratio': bg_ratio})
            segment_dict_list = segment_dict_list_temp
            # print(segment_dict_list)
            if line_num >= 2:
                # get blank segment height for line segmentation
                blank_dict_list = []
                max_height = -1
                for segment_idx in range(len(segment_dict_list)):
                    if len(blank_dict_list) != 0:
                        blank_dict_list[-1]['end'] = segment_dict_list[segment_idx]['start']
                        blank_dict_list[-1]['height'] = blank_dict_list[-1]['end'] - blank_dict_list[-1]['start']
                        if blank_dict_list[-1]['height'] > max_height:
                            max_height = blank_dict_list[-1]['height']
                    blank_dict_list.append({'start': segment_dict_list[segment_idx]['end']})
                blank_dict_list.pop()
                # print(blank_dict_list)
                # pick blank segment that is long enough
                crop_height_list = []
                for blank_dict in blank_dict_list:
                    if blank_dict['height'] >= max_height - 2:
                        crop_height_list.append((blank_dict['start'] + blank_dict['end']) / 2)
                crop_height_list.append(image_tuple[0].height)
            else:
                # blank_dict_list = None
                # just one line of equation
                crop_height_list = [image_tuple[0].height]
            """
            print(segment_dict_list)
            print(blank_dict_list)
            print(crop_height_list)
            """
            accum_height = 0
            pos_per_pixel = (image_tuple[1]['bottom'] - image_tuple[1]['top']) / image_tuple[0].height
            for crop_height in crop_height_list:
                image_dict_list.append(
                    {'image': image_tuple[0].crop((0, accum_height, image_tuple[0].width, crop_height)),
                     'x0': image_tuple[1]['x0'],
                     'x1': image_tuple[1]['x1'],
                     'top': image_tuple[1]['top'] + accum_height * pos_per_pixel,
                     'doctop': image_tuple[1]['doctop'] + accum_height * pos_per_pixel,
                     'bottom': image_tuple[1]['bottom'] - (image_tuple[0].height - crop_height) * pos_per_pixel, })
                accum_height = crop_height

        converted_images = []
        for image_dict in image_dict_list:
            ocr_result = self.ocr_model(image_dict['image'])
            ocr_result = re.sub(r'qquad(\\qquad)+', 'qquad', ocr_result)
            ocr_result = re.sub(r'[^\\]\\([!,:; ]|quad)', ' ', ocr_result)
            converted_images.append({'text': ocr_result + '\n',
                                     'x0': image_dict['x0'],
                                     'x1': image_dict['x1'],
                                     'top': image_dict['top'],
                                     'doctop': image_dict['doctop'],
                                     'bottom': image_dict['bottom'],
                                     'upright': True,
                                     'direction': 1,
                                     'size': 0,
                                     'fontname': 'Image'})

        return converted_images
